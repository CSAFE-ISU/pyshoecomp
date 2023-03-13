# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from skimage import io as skio
from skimage import util as skutil
from skimage import transform as sktrans


class ImageDesc:
    def __init__(self, raw_img, name: str = "<unk>", filename: Optional[str] = None):
        self.img = raw_img
        self.name = name
        self.filename = filename

    @classmethod
    def from_file(
        cls,
        filepath: str,
        scale_factor: float = 1.0,
        outer_crop: int = 0,
        flip=False
    ):
        name = os.path.splitext(os.path.basename(filepath))[0]
        img = skio.imread(filepath, as_gray=True)
        if scale_factor != 1.0:
            img = sktrans.rescale(
                img, scale_factor, mode="symmetric", anti_aliasing=True
            )
        if outer_crop > 0:
            img = skutil.crop(img, outer_crop, copy=True, order="C")
        if flip:
            img = np.flip(img, axis=1)
        img = np.float32(img)
        return ImageDesc(raw_img=img, name=name, filename=filepath)
