import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

from ..ndarray.utils import first_nonzero, last_nonzero
# from ..ocr.text.extraction import extract_text
# from ..ocr.character.extraction import extract_characters


class Strip(object):
    ''' Strip image.'''

    def __init__(self, image, index, mask=None):

        h, w = image.shape[: 2]
        if mask is None:
            mask = 255 * np.ones((h, w), dtype=np.uint8)

        self.h = h
        self.w = w
        self.image = cv2.bitwise_and(image, image, mask=mask)
        self.index = index
        self.mask = mask

        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def shift(self, disp):
        ''' shift strip vertically. '''

        M = np.float32([[1, 0, 0], [0, 1, disp]])
        self.image = cv2.warpAffine(self.image, M, (self.w, self.h))
        self.mask = cv2.warpAffine(self.mask, M, (self.w, self.h))
        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
        return self


    def filled_image(self):
        ''' Return image with masked-out areas in white. '''

        return cv2.bitwise_or(
            self.image, cv2.cvtColor(
                cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB
            )
        )


    def is_blank(self, blank_tresh=127):
        ''' Check whether is a blank strip. '''

        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0


    def stack(self, other, disp=0, filled=False):
        ''' Stack horizontally with other strip. '''

        y1_min, y1_max = 0, self.h - 1
        y2_min, y2_max = 0, other.h - 1
        y_inter_min = max(0, disp)
        y_inter_max = min(y1_max, y2_max + disp) + 1
        h_inter = y_inter_max - y_inter_min

        # borders coordinates
        r1 = self.offsets_r[y_inter_min : y_inter_max]

        if disp >= 0:
            l2 = other.offsets_l[: h_inter]
        else:
            l2 = other.offsets_l[-disp : -disp + h_inter]

        # horizontal offset
        offset = self.w - np.min(l2 + self.w - r1) + 1

        # union
        y_union_min = min(0, disp)
        y_union_max = max(y1_max, y2_max + disp) + 1
        h_union = y_union_max - y_union_min

        min_h, max_h = min(self.h, other.h), max(self.h, other.h)

        # new image / mask
        temp_image = np.zeros((h_union, offset + other.w, 3), dtype=np.uint8)
        temp_mask = np.zeros((h_union, offset + other.w), dtype=np.uint8)
        if disp >= 0:
            temp_image[: self.h, : self.w] = self.image
            temp_image[disp : disp + other.h, offset :] += other.image
            temp_mask[: self.h, : self.w] = self.mask
            temp_mask[disp : disp + other.h, offset :] += other.mask
        else:
            temp_image[-disp : -disp + self.h, : self.w] = self.image
            temp_image[: other.h, offset :] += other.image
            temp_mask[-disp : -disp + self.h, : self.w] = self.mask
            temp_mask[: other.h, offset :] += other.mask

        self.h, self.w = temp_mask.shape
        self.image = temp_image
        self.mask = temp_mask
        self.offsets_l =np.apply_along_axis(first_nonzero, 1, self.mask)
        self.offsets_r =np.apply_along_axis(last_nonzero, 1, self.mask)
        if filled:
            self.image = self.filled_image()
        return self


    def crop_vertically(self, y1, y2):
        ''' Crop the strip vertically from h1 to h2. '''

        self.offsets_l = self.offsets_l[y1 : y2 + 1]
        self.offsets_r = self.offsets_r[y1 : y2 + 1]
        x1 = self.offsets_l.min()
        x2 = self.offsets_r.max()
        self.offsets_l -= x1
        self.offsets_r -= x1
        self.image = self.image[y1 : y2 + 1, x1 : x2 + 1] # height can be different from y2 - y1 for the bottom part of the document
        self.mask = self.mask[y1 : y2 + 1, x1 : x2 + 1]
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
        self.w = self.mask.shape[1]
        self.h = self.mask.shape[0]