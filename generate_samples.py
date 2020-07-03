import os
import numpy as np
import cv2
import argparse
import math
import random
import glob
import time


''' Command line
$ python generate_samples.py --save-dir [path]
Elapsed time=5.03 minutes (302.0571186542511 seconds)
'''

'''
ISRI-OCR Tk repository
https://code.google.com/archive/p/isri-ocr-evaluation-tools
'''
ISRI_DATASET_DIR = 'datasets/isri-ocr'

# ignore images of the test set (dataset D2)
ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
                 '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
                 '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
                 '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']


def init(args):
    ''' Initial setup. '''

    # build tree directory
    samples_dir = os.path.expanduser(args.samples_dir)
    os.makedirs('{}/positives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/positives/val'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/val'.format(samples_dir), exist_ok=True)


def generate_samples(args):
    ''' Sampling process. '''

    samples_dir = os.path.expanduser(args.samples_dir)
    if glob.glob('{}/**/*.jpg'.format(samples_dir)):
        print('Sampling already done!')
        return

    docs = glob.glob('{}/**/*.tif'.format(ISRI_DATASET_DIR), recursive=True)

    # filter documents in ignore list
    docs = [doc for doc in docs if os.path.basename(doc).replace('.tif', '') not in ignore_images]
    random.shuffle(docs)
    if args.num_docs is not None:
        docs = docs[ : args.num_docs]

    # split train and val sets
    num_docs = len(docs)
    docs_train = docs[int(args.ratio_val * num_docs) :]
    docs_val = docs[ : int(args.ratio_val * num_docs)]

    processed = 0
    size_right = math.ceil(args.input_size / 2)
    size_left = args.input_size - size_right
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}#, 'neutral': 0}
        for doc in docs:
            max_per_doc = 0

            print('Processing document {}/{}[mode={}]'.format(processed + 1, num_docs, mode))
            processed += 1

            # shredding
            print('     => Shredding')
            image = cv2.imread(doc)
            h, w, c = image.shape
            acc = 0
            strips = []
            for i in range(args.num_strips):
                dw = int((w - acc) / (args.num_strips - i))
                strip = image[:, acc : acc + dw]
                noise_left = np.random.randint(0, 255, (h, args.disp_noise)).astype(np.uint8)
                noise_right = np.random.randint(0, 255, (h, args.disp_noise)).astype(np.uint8)
                for j in range(c): # for each channel
                    strip[:, : args.disp_noise, j] = cv2.add(strip[:, : args.disp_noise, j], noise_left)
                    strip[:, -args.disp_noise :, j] = cv2.add(strip[:, -args.disp_noise :, j], noise_right)
                strips.append(strip)
                acc += dw

            # positives
            print('     => Positive samples')
            N = len(strips)
            combs = [(i, i + 1) for i in range(N - 1)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                print('[{}][{}] :: total={}'.format(i, j, count['positives']))
                image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                for y in range(0, h - args.input_size, args.stride):
                    crop = image[y : y + args.input_size]
                    if (crop != 255).sum() / crop.size >= args.thresh_black:
                        count['positives'] += 1
                        max_per_doc += 1
                        cv2.imwrite('{}/positives/{}/{}.jpg'.format(samples_dir, mode, count['positives']), crop)
                        if max_per_doc == args.max_pos:
                            stop = True
                            break
                if stop:
                    break


            print('     => Negative samples')
            # negatives
            combs = [(i, j) for i in range(N) for j in range(N) if (i != j) and (i + 1 != j)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                print('[{}][{}] :: total={}'.format(i, j, count['negatives']))
                image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                for y in range(0, h - args.input_size, args.stride):
                    crop = image[y : y + args.input_size]
                    if (crop != 255).sum() / crop.size >= args.thresh_black:
                        count['negatives'] += 1
                        cv2.imwrite('{}/negatives/{}/{}.jpg'.format(samples_dir, mode, count['negatives']), crop)
                        if count['negatives'] >= int(args.ratio_neg * count['positives']):
                            stop = True
                            break
                if stop:
                    break


def generate_txt(args):
    ''' Generate train.txt and val.txt. '''

    samples_dir = os.path.expanduser(args.samples_dir)
    docs_neg_train = glob.glob('{}/negatives/train/*.jpg'.format(samples_dir))
    docs_neg_val = glob.glob('{}/negatives/val/*.jpg'.format(samples_dir))
    docs_pos_train = glob.glob('{}/positives/train/*.jpg'.format(samples_dir))
    docs_pos_val = glob.glob('{}/positives/val/*.jpg'.format(samples_dir))

    neg_train = ['{} 0'.format(doc) for doc in docs_neg_train]
    pos_train = ['{} 1'.format(doc) for doc in docs_pos_train]
    neg_val = ['{} 0'.format(doc) for doc in docs_neg_val]
    pos_val = ['{} 1'.format(doc) for doc in docs_pos_val]

    train = neg_train + pos_train
    val = neg_val + pos_val
    random.shuffle(train)
    random.shuffle(val)

    # save
    open('{}/train.txt'.format(samples_dir), 'w').write('\n'.join(train))
    open('{}/val.txt'.format(samples_dir), 'w').write('\n'.join(val))


def main():

    parser = argparse.ArgumentParser(description='DeepRec (SIB18) :: Generation of local samples for training.')
    parser.add_argument(
        '-rv', '--ratio-val', action='store', dest='ratio_val', required=False, type=float,
        default=0.1, help='Ratio between the number of samples reserved for validation and the total.'
    )
    parser.add_argument(
        '-i', '--input-size', action='store', dest='input_size', required=False, type=int,
        default=31, help='Sample input size.'
    )
    parser.add_argument(
        '-tb', '--thresh-black', action='store', dest='thresh_black', required=False, type=float,
        default=0.2, help='Ratio threshold of black pixels in a sample..'
    )
    parser.add_argument(
        '-d', '--disp-noise', action='store', dest='disp_noise', required=False, type=int,
        default=2, help='Displacement from edge where noise will be applied.'
    )
    parser.add_argument(
        '-m', '--max-pos', action='store', dest='max_pos', required=False, type=int,
        default=1000, help='Max. positives per document.'
    )
    parser.add_argument(
        '-rn', '--ratio-neg', action='store', dest='ratio_neg', required=False, type=float,
        default=1.0, help='Ratio between number of negative samples and positives.'
    )
    parser.add_argument(
        '-nd', '--num-docs', action='store', dest='num_docs', required=False, type=int,
        default=None, help='Number of documents.'
    )
    parser.add_argument(
        '-ns', '--num-strips', action='store', dest='num_strips', required=False, type=int,
        default=30, help='Number of strips generated for each document.'
    )
    parser.add_argument(
        '-se', '--seed', action='store', dest='seed', required=False, type=float,
        default=0.0, help='Seed (float) for the training process.'
    )
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/samples/deeprec-sib18', help='Path where the generated samples will be placed.'
    )
    parser.add_argument(
        '-st', '--stride', action='store', dest='stride', required=False, type=int,
        default=2, help='Vertical stride for consecutive samples.'
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    init(args)
    print('Extracting characters')
    generate_samples(args)
    print('Generate txt files')
    generate_txt(args)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
