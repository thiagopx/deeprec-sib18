import os
import re
import numpy as np
import cv2
import random
import copy

from .strip import Strip


class Strips(object):
    ''' Strips operations manager.'''

    def __init__(self, path=None, filter_blanks=True, blank_tresh=127):
        ''' Strips constructor.

        @path: path to a directory containing strips (in case of load real strips)
        @strips_list: list of strips (objects of Strip class)
        @filter_blanks: true-or-false flag indicating the removal of blank strips
        @blank_thresh: threshold used in the blank strips filtering
        '''

        assert (path is not None) or (strips_list is not None)


        self.strips = []
        self.artificial_mask = False
        if path is not None:
            assert os.path.exists(path)
            self._load_data(path)
        else:
            self.strips = [strip.copy() for strip in strips_list]

        # remove low content ('blank') strips
        if filter_blanks:
            self.strips = [strip for strip in self.strips if not strip.is_blank(blank_tresh)]
            new_indices = np.argsort([strip.index for strip in self.strips])
            for strip, new_index in zip(self.strips, new_indices):
                strip.index = int(new_index) # avoid json serialization issues

        self.left_extremities = [self(0)]
        self.right_extremities = [self(-1)]


    def __call__(self, i):
        ''' Returns the i-th strip. '''

        return self.strips[i]


    def __add__(self, other):
        ''' Including new strips. '''

        N = len(self.strips)
        union = self.copy()
        other = other.copy()
        for strip in other.strips:
            strip.index += N

        union.left_extremities += other.left_extremities
        union.right_extremities += other.right_extremities
        union.strips += other.strips
        return union


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def size(self):
        ''' Number of strips. '''

        return len(self.strips)


    def shuffle(self):

        random.shuffle(self.strips)
        return self


    def permutation(self):
        ''' Return the permutation (order) of the strips. '''

        return [strip.index for strip in self.strips]


    def extremities(self):
        ''' Return the ground-truth indices of the strips belonging to the documents' extremities. '''

        left_indices = [strip.index for strip in self.left_extremities]
        right_indices = [strip.index for strip in self.right_extremities]
        return left_indices, right_indices


    def _load_data(self, path, regex_str='.*\d\d\d\d\d\.*'):
        ''' Stack strips horizontally.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        '''

        path_images = '{}/strips'.format(path)
        path_masks = '{}/masks'.format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted([fname for fname in os.listdir(path_images) if regex.match(fname)])
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread('{}/{}'.format(path_images, fname)),
                cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load('{}/{}.npy'.format(path_masks, os.path.splitext(fname)[0]))
                masks.append(mask)
        else:
            masks = len(images) * [None]
            self.artificial_mask = True

        for index, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, index, mask)
            self.strips.append(strip)


    def image(self, order=None, ground_truth_order=False, displacements=None, filled=False):
        ''' Return the document as an image.

        order: list with indices of the strips (if not None, ground_truth order is ignored).
        ground_truth_order: if True (and order is None), it composes the image in the ground-truth order.
        displacements: relative displacements between neighbors strips.
        filled: if True, the background is white.
        '''

        corrected = []
        if order is None:
            if ground_truth_order:
                corrected = sorted(self.strips, key=lambda x: x.index)
            else:
                corrected = self.strips
        else:
            corrected = [self(idx) for idx in order]

        if displacements is None:
            displacements = len(self.strips) * [0]
        stacked = corrected[0].copy()
        for current, disp in zip(corrected[1 :], displacements):
            stacked.stack(current, disp=disp, filled=filled)
        return stacked.image


    def post_processed_image(self, order=None, ground_truth_order=False, displacements=None, filled=False, delta_y=50):

        # extracting crops
        h = self.image().shape[0]
        crops = []
        for y in range(0, h - delta_y, delta_y):
            crop = self.copy().crop_vertically(y, y + delta_y - 1).image(
                order=order, ground_truth_order=ground_truth_order,
                displacements=displacements, filled=filled
            )
            crops.append(crop)

        # result image
        h = sum([crop.shape[0] for crop in crops])
        w = max([crop.shape[1] for crop in crops])
        result = np.empty((h, w, 3), dtype=np.uint8)
        result[:] = 255 if filled else 0

        # joining crops
        y = 0
        for crop in crops:
            w = crop.shape[1]
            result[y : y + delta_y, : w] = crop
            y += delta_y
        return result


    def crop_vertically(self, y1, y2):
        ''' Crop the strips vertically from y1 to y2. '''
        i = 0
        for strip in self.strips:
            i += 1
            strip.crop_vertically(y1, y2)
        return self