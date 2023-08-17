# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage import io as skio
from skimage import util as skutil
from skimage import transform as sktrans

#
from _reconfig import Config


class ImageDesc:
    def __init__(self, raw_img, name="<unk>", filename=None):
        self.img = raw_img
        self.name = name
        self.filename = filename

    @classmethod
    def _from_file(
        cls,
        filepath: str,
        scale,
        crop,
        x_flip=False,
        y_flip=False,
    ):
        name = os.path.splitext(os.path.basename(filepath))[0]
        img = skio.imread(filepath, as_gray=True)
        if scale != 1.0:
            img = sktrans.rescale(img, scale, mode="symmetric", anti_aliasing=True)
        if isinstance(crop, int) or isinstance(crop, float):
            cfinal = ((crop, crop), (crop, crop))
        elif isinstance(crop, tuple):
            cfinal = crop
        else:
            raise RuntimeError(f"invalid crop {crop}")
        img = skutil.crop(img, cfinal, copy=True, order="C")
        if y_flip:
            img = np.flip(img, axis=0)
        if x_flip:
            img = np.flip(img, axis=1)
        img = np.float32(img)
        return ImageDesc(raw_img=img, name=name, filename=filepath)

    @classmethod
    def from_file(cls, filepath, is_k, is_match):
        na1 = "img_K" if is_k else "img_Q"
        na2 = "1" if is_match else "0"
        name = na1 + na2
        return cls._from_file(filepath, **Config.get_params(name))
