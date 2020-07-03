import numpy as np
import os
import cv2
import math
import argparse
import tensorflow as tf
import json
from skimage.filters import threshold_otsu

from docrec.strips.strips import Strips
from docrec.models.squeezenet import SqueezeNet
from docrec.models.mobilenet import MobileNetFC


def extract_features(strip, input_size):
    ''' Extract image around the border. '''

    input_size_h, input_size_w = input_size
    image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(image)
    thresholded = (image > thresh).astype(np.float32)

    image_bin = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last

    wl = math.ceil(input_size_w / 2)
    wr = input_size_w - wl
    h, w, _ = strip.image.shape
    offset = int((h - input_size_h) / 2)

    # left image
    left_border = strip.offsets_l + 2
    left = np.ones((input_size_h, wl, 3), dtype=np.float32)
    for y, x in enumerate(left_border[offset : offset + input_size_h]):
        w_new = min(wl, w - x)
        left[y, : w_new] = image_bin[y + offset, x : x + w_new]

    # right image
    right_border = strip.offsets_r - 2
    right = np.ones((input_size_h, wr, 3), dtype=np.float32)
    for y, x in enumerate(right_border[offset : offset + input_size_h]):
        w_new = min(wr, x + 1)
        right[y, : w_new] = image_bin[y + offset, x - w_new + 1: x + 1]

    return left, right


def create_overlay(strip_left, strip_right, conv10, input_size):

    # input_size = tuple(args.input_size)
    input_size_h, input_size_w = input_size
    wr = input_size_w // 2
    neg = conv10[:, :, 0]
    pos = conv10[:, :, 1]

    maps = np.stack([0 * pos, pos, neg]) # BGR
    maps = maps / maps.max()

    maps = (255 * np.transpose(maps, axes=(1, 2, 0))).astype(np.uint8) # channels last
    maps = cv2.resize(maps, dsize=(wr, input_size_h), interpolation=cv2.INTER_CUBIC)

    # left strip
    overlay_left = strip_left.image.copy()
    offset = (strip_left.h - input_size_h) // 2
    left_border, right_border = strip_left.offsets_l[offset : offset + input_size_h], strip_left.offsets_r[offset : offset + input_size_h]
    for y, (x1, x2) in enumerate(zip(left_border, right_border)):
        wr_ = min(x2 - x1 + 1, wr)
        overlay_left[y + offset, x2 - wr_ + 1 : x2 + 1] = maps[y, : wr_]

    # right strip
    overlay_right = strip_right.image.copy()
    offset = (strip_right.h - input_size_h) // 2
    left_border, right_border = strip_right.offsets_l[offset : offset + input_size_h], strip_right.offsets_r[offset : offset + input_size_h]
    for y, (x1, x2) in enumerate(zip(left_border, right_border)):
        wr_ = min(x2 - x1 + 1, wr)
        overlay_right[y + offset, x1 : x1 + wr_] = maps[y, : wr_]

    return overlay_left, overlay_right


def binary(strip):

    strip = strip.copy()
    image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(image)
    thresholded = (255 * (image > thresh)).astype(np.uint8)
    thresholded = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last
    strip.image = cv2.bitwise_and(
        thresholded, cv2.cvtColor(strip.mask, cv2.COLOR_GRAY2RGB)
    )
    return strip


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepRec (SIB18) :: Pairwise compatibility heatmap.')
    parser.add_argument(
        '-d', '--doc', action='store', dest='doc', required=False, type=str,
        default='datasets/D1/artificial/D001', help='Document.'
    )
    parser.add_argument(
        '-ar', '--arch', action='store', dest='arch', required=False, type=str,
        default='squeezenet', help='Architecture of the CNN'
    )
    parser.add_argument(
        '-al', '--alpha', action='store', dest='alpha', required=False, type=float,
        default=0.5, help='Alpha channel.'
    )
    parser.add_argument(
        '-vs', '--vshift', action='store', dest='vshift', required=False, type=int,
        default=10, help='Vertical shift.'
    )
    args = parser.parse_args()

    assert args.arch in ['squeezenet', 'mobilenet']

    # data
    input_size_h = 3000
    input_size_w = 31 if args.arch == 'squeezenet' else 32
    wl = math.ceil(input_size_w / 2)
    wr = input_size_w - wl
    images_ph = tf.placeholder(
        tf.float32, name='images_ph', shape=(None, input_size_h, input_size_w, 3) # channels last
    )
    batch = np.ones((2 * args.vshift + 1, input_size_h, input_size_w, 3), dtype=np.float32)

    # architecture definition
    if args.arch == 'squeezenet':
        model = SqueezeNet(images_ph, num_classes=2, mode='val', channels_first=False)
        model_file_ext = 'npy'
    else:
        model = MobileNetFC(input_tensor=images_ph, num_classes=2)
        model_file_ext = 'h5'
    sess = model.sess
    logits_op = model.output
    conv10_op = model.view

    probs_op = tf.nn.softmax(logits_op)
    comp_op = tf.reduce_max(probs_op[:, 1])
    disp_op = tf.argmax(probs_op[:, 1]) - args.vshift

    sess.run(tf.global_variables_initializer())
    best_epoch = json.load(open('traindata/{}/info.json'.format(args.arch), 'r'))['best_epoch']
    weights_path = 'traindata/{}/model/{}.{}'.format(args.arch, best_epoch, model_file_ext)
    model.load_weights(weights_path)

    base_path = 'illustration/heatmap/{}'.format(args.arch)
    os.makedirs(base_path, exist_ok=True)

    strips_regular = Strips(path=args.doc, filter_blanks=True)
    strips_shuffled = strips_regular.copy().shuffle()

    N = strips_regular.size()
    for strips, strips_type in zip([strips_regular, strips_shuffled], ['regular', 'shuffled']):

        # features
        features = []
        for strip in strips.strips:
            left, right = extract_features(strip, (input_size_h, input_size_w))
            features.append((left, right))

        # converto to BGR
        for strip in strips.strips:
            strip.image = strip.image[..., :: - 1]

        for i in range(N - 1):
            batch[:, :, : wr] = features[i][1]
            batch[args.vshift, :, wr :, :] = features[i + 1][0] # radius zero
            for r in range(1, args.vshift + 1):
                batch[args.vshift - r, : -r, wr :] = features[i + 1][0][r :]  # slide up
                batch[args.vshift + r, r : , wr :] = features[i + 1][0][: -r] # slide down

            probs, conv10, comp, disp = sess.run([probs_op, conv10_op, comp_op, disp_op], feed_dict={images_ph: batch})
            batch[:] = 1.0 # reset
            overlay_left, overlay_right = create_overlay(
                strips(i), strips(i + 1),
                conv10[disp + args.vshift],
                (input_size_h, input_size_w)
            )
            strips(i).image = cv2.addWeighted(overlay_left, args.alpha, strips(i).image, 1 - args.alpha, 0)
            strips(i + 1).image = cv2.addWeighted(overlay_right, args.alpha, strips(i + 1).image, 1 - args.alpha, 0)
        strip = strips(0).copy()
        for i in range(1, N):
            strip.stack(strips(i), filled=False)

        _, dataset, _, doc = args.doc.split('/')
        cv2.imwrite('{}/{}-{}_{}.jpg'.format(base_path, dataset, doc, strips_type), strip.image)
    sess.close()