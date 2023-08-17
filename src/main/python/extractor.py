# - * - coding : utf - 8 - * -
import numpy as np
from skimage.feature import ORB, CENSURE, corner_fast, corner_peaks

# config
from _reconfig import Config


def uniqueify(f):
    def g(*args, **kwargs):
        res = f(*args, **kwargs)
        ans = np.unique(res, axis=0)
        return ans

    return g


class Extractor:
    _extname_ = "<none>"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, img):
        """
        receive an image (grayscale) => return interest points
        interest points must be (row,column)
        """
        raise NotImplementedError("abstract base class")


class ORBExtractor(Extractor):
    _extname_ = "ORB"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = ORB(**Config.get_params("ORB"))

    @uniqueify
    def __call__(self, img):
        self.etor.detect(img)
        return self.etor.keypoints


class CENSUREExtractor(Extractor):
    _extname_ = "CENSURE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = CENSURE(**Config.get_params("CENSURE"))

    @uniqueify
    def __call__(self, img):
        self.etor.detect(img)
        return self.etor.keypoints


class FastExtractor(Extractor):
    _extname_ = "FAST"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = lambda img: corner_peaks(
            corner_fast(img, **Config.get_params("FAST_params")),
            **Config.get_params("FAST_peaks")
        )

    @uniqueify
    def __call__(self, img):
        keypoints = self.etor(img)
        return keypoints


EXTRACTOR_MAP = {x._extname_: x for x in Extractor.__subclasses__()}
