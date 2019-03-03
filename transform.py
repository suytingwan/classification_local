from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps

import numpy as np
import numbers
import types
import collections
import warnings

import functional as F

def scale(img, size, interpolation):
    # scale
    w, h = img.size

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            image_new = img
        if w < h:
            ow = size
            oh = int(size * h / w)
            img_new = img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            image_new = img.resize((ow, oh), interpolation)
    else:
        image_new = img.resize(size, interpolation)
    return image_new


class RandomSizedCrop(object):
    '''
    Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
    '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections))
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        area = h * w

        attempt = 0
        while attempt < 10:
            targetArea = random.uniform(0.08, 1.0) * area
            aspectRatio = random.uniform(3/4, 4/3)

            w_new = torch.round(math.sqrt(targetArea * aspectRatio))
            h_new = torch.round(math.sqrt(targetArea / aspectRatio))

            if random.random() < 0.5:
                w_new, h_new = h_new, w_new

            if h_new < h and w_new < w:
                y1 = random.randint(0, h - h_new)
                x1 = random.randint(0, w - w_new)

                image_new = img.crop(x1, y1, x1+w_new, y1+h_new)

                return scale(image_new, self.size, self.interpolation)
            else:
                attempt += 1

        # scale
        image_new = scale(img, self.size, self.interpolation)

        #center crop
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return image_new.crop((x1, y1, x1 + tw, y1 + th))



class RandomVerticalFlip(object):

    def __call__(self, img):
        '''
        :param img (PIL.Image): Image to be flipped.
        :return: PIL.Image: Randomly flipped image
        '''

        if random.random < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.hue = 0

    def __call__(self, img):
        transforms = []

        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform(img)


class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of the len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])

        return F.rotate(img, angle, self.resample, self.expand, self.center)

class RandomAffine(object):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, img):

        angle = random.randint(self.degrees[0], self.degrees[1])
        img_size = img.size
        if self.translate is not None:
            max_dx = self.translate[0] * img_size[0]
            max_dy = self.translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.shear is not None:
            shear = random.uniform(self.shear[0], self.shear[1])
        else:
            shear = 0.0

        return F.affine(img, angle, translations, scale, shear, resample=self.resample, fillcolor=self.fillcolor)

class FiveCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h,w) for size"
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)
    
class TenCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h,w) for size"
            self.size = size
    def __call__(self, img):
        return F.ten_crop(img, self.size)

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
    
    def __call__(self, img):
        return self.lambd(img)








