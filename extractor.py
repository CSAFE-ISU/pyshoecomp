# - * - coding : utf - 8 - * -
from typing import Optional, Any
import numpy as np
from numpy.typing import ArrayLike

# mixing opencv and skimage may be NASTY
# but let's do it for now
from skimage.feature import SIFT, ORB, CENSURE, corner_fast, corner_peaks
import cv2

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

    def __call__(self, img: ArrayLike) -> ArrayLike:
        """
        receive an image (grayscale) => return interest points
        interest points must be (row,column)
        """
        raise NotImplementedError("abstract base class")


class SIFTExtractor(Extractor):
    _extname_ = "SIFT"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = SIFT(**Config.get_params("SIFT"))

    @uniqueify
    def __call__(self, img):
        self.etor.detect(img)
        return self.etor.keypoints


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


class TomasiExtractor(Extractor):
    _extname_ = "Shi-Tomasi"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = dict(**Config.get_params("Shi-Tomasi"))

    @uniqueify
    def __call__(self, img):
        corners = cv2.goodFeaturesToTrack(image=img, **self.params)
        # corners are x, y format, so convert to row / column
        return corners[:, 0, ::-1]


class KAZEExtractor(Extractor):
    _extname_ = "KAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = cv2.KAZE_create(**Config.get_params("KAZE"))

    @uniqueify
    def __call__(self, img):
        keypoints = self.etor.detect(img)
        keypoints = np.array([k.pt[::-1] for k in keypoints])
        return keypoints


class AKAZEExtractor(Extractor):
    _extname_ = "AKAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor = cv2.AKAZE_create(**Config.get_params("AKAZE"))

    @uniqueify
    def __call__(self, img):
        keypoints = self.etor.detect(img)
        keypoints = np.array([k.pt[::-1] for k in keypoints])
        return keypoints


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


class SIFT_ORB(Extractor):
    _extname_ = "SIFT+ORB"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = ORB(**Config.get_params("ORB"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        self.etor2.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.keypoints
        return np.vstack((kp2, kp1))


class SIFT_CENSURE(Extractor):
    _extname_ = "SIFT+CENSURE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = CENSURE(**Config.get_params("CENSURE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        self.etor2.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.keypoints
        return np.vstack((kp2, kp1))


class SIFT_Fast(Extractor):
    _extname_ = "SIFT+FAST"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = lambda img: corner_peaks(
            corner_fast(img, **Config.get_params("FAST_params")),
            **Config.get_params("FAST_peaks")
        )

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2(img)
        return np.vstack((kp2, kp1))


class SIFT_KAZE(Extractor):
    _extname_ = "SIFT+KAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = cv2.KAZE_create(**Config.get_params("KAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.detect(img)
        kp2 = np.array([k.pt[::-1] for k in kp2])
        return np.vstack((kp2, kp1))


class SIFT_AKAZE(Extractor):
    _extname_ = "SIFT+AKAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = cv2.AKAZE_create(**Config.get_params("AKAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.detect(img)
        kp2 = np.array([k.pt[::-1] for k in kp2])
        return np.vstack((kp2, kp1))


class SIFT_Fast_KAZE(Extractor):
    _extname_ = "SIFT+Fast+KAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = lambda img: corner_peaks(
            corner_fast(img, **Config.get_params("FAST_params")),
            **Config.get_params("FAST_peaks")
        )
        self.etor3 = cv2.KAZE_create(**Config.get_params("KAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2(img)
        kp3 = self.etor3.detect(img)
        kp3 = np.array([k.pt[::-1] for k in kp3])
        return np.vstack((kp3, kp2, kp1))


class SIFT_Fast_AKAZE(Extractor):
    _extname_ = "SIFT+Fast+AKAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = lambda img: corner_peaks(
            corner_fast(img, **Config.get_params("FAST_params")),
            **Config.get_params("FAST_peaks")
        )
        self.etor3 = cv2.AKAZE_create(**Config.get_params("AKAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2(img)
        kp3 = self.etor3.detect(img)
        kp3 = np.array([k.pt[::-1] for k in kp3])
        return np.vstack((kp3, kp2, kp1))


class SIFT_ORB_KAZE(Extractor):
    _extname_ = "SIFT+ORB+KAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = ORB(**Config.get_params("ORB"))
        self.etor3 = cv2.KAZE_create(**Config.get_params("KAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        self.etor2.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.keypoints
        kp3 = self.etor3.detect(img)
        kp3 = np.array([k.pt[::-1] for k in kp3])
        return np.vstack((kp3, kp2, kp1))


class SIFT_ORB_AKAZE(Extractor):
    _extname_ = "SIFT+ORB+AKAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = ORB(**Config.get_params("ORB"))
        self.etor3 = cv2.AKAZE_create(**Config.get_params("AKAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        self.etor2.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.keypoints
        kp3 = self.etor3.detect(img)
        kp3 = np.array([k.pt[::-1] for k in kp3])
        return np.vstack((kp3, kp2, kp1))


class SIFT_KAZE_AKAZE(Extractor):
    _extname_ = "SIFT+KAZE+AKAZE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etor1 = SIFT(**Config.get_params("SIFT"))
        self.etor2 = cv2.KAZE_create(**Config.get_params("KAZE"))
        self.etor3 = cv2.AKAZE_create(**Config.get_params("AKAZE"))

    @uniqueify
    def __call__(self, img):
        self.etor1.detect(img)
        kp1 = self.etor1.keypoints
        kp2 = self.etor2.detect(img)
        kp2 = np.array([k.pt[::-1] for k in kp2])
        kp3 = self.etor3.detect(img)
        kp3 = np.array([k.pt[::-1] for k in kp3])
        return np.vstack((kp3, kp2, kp1))


EXTRACTOR_MAP = {x._extname_: x for x in Extractor.__subclasses__()}
