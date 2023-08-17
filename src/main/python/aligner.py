from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP
from corresponder import CliqueMatcher

import warnings
import skimage.transform as sktrans
import numpy as np
import cliquematch
import time

# REMEMBER skimage warp IS WHACK
# REMEMBER MAP_FUNC IS FROM K TO Q !!!
# REMEMBER MAP_FUNC needs coordinates in (X, Y)
# and you have them in (row, col) !!!!!!


def centerify(x):
    cent = np.mean(x, axis=0)
    return cent, x - cent


class AlignFunction:
    _extname_ = "<none>"

    def __init__(self, *args, **params):
        pass

    def _get_mapping(self, Q, K, corr, *args, **params):
        """
        receive Q, K, and correspondence info
        return something that can go into sktrans.warp
        """
        raise NotImplementedError("abstract base class")

    def __call__(self, Q, K, corr, *args, **params):
        return self._get_mapping(Q, K, corr, *args, **params)

    def align_Q_to_K(self, Q, K, corr, *args, **params):
        map_func = params.get(
            "map_func", self._get_mapping(Q, K, corr, *args, **params)
        )
        return sktrans.warp(
            Q.img,
            inverse_map=map_func,
            output_shape=K.img.shape,
            mode="constant",
            cval=1,
        )


class DummyMapping(AlignFunction):
    _extname_ = "dummy"

    def _get_mapping(self, Q, K, corr, *args, **params):
        return lambda x: x


class KabschMapping(AlignFunction):
    _extname_ = "kabsch"

    def _get_mapping(self, Q, K, corr, *args, **params):
        # why are we swapping to (col, row)?
        # because skimage.transform.warp is WHACK
        # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
        Q_pts = corr["Q"][:, ::-1]
        K_pts = corr["K"][:, ::-1]
        # kabsch algorithm to get rotation and translation
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        Q_cent, Q_norm = centerify(Q_pts)
        K_cent, K_norm = centerify(K_pts)

        # why are we mapping K to Q?
        # skimage.transform.warp is WHACK,
        # it requires the INVERSE MAP
        H = np.matmul(K_norm.T, Q_norm)
        V, s, U = np.linalg.svd(H, full_matrices=True, compute_uv=True)

        rotmat = np.eye(2, 2)
        d = np.linalg.det(np.matmul(V, U.T))
        rotmat[1, 1] = 1 if d > 0 else -1
        rotmat = np.matmul(np.matmul(V, rotmat), U.T)

        dx, dy = -np.matmul(K_cent, rotmat) + Q_cent
        if np.isnan(rotmat[0, 0]):
            return lambda x: x
        theta = np.arccos(rotmat[0, 0])
        theta = -theta if rotmat[1, 0] < 0 else theta
        theta = 0 if np.isnan(theta) or np.isinf(theta) else theta

        # print("theta is", theta)
        map_func = sktrans.EuclideanTransform(
            # using -theta because counterclockwise?
            rotation=-theta,
            translation=(dx, dy),
        )
        return map_func


class PolynomialMapping(AlignFunction):
    _extname_ = "polynomial"

    def __init__(self, order, *args, **params):
        if order not in (2, 3, 4):
            raise RuntimeError("invalid polynomial order")
        self.order = order
        self._extname_ = f"polynomial{order}"

    def _get_mapping(self, Q, K, corr, *args, **params):
        Q_pts = corr["Q"][:, ::-1]
        K_pts = corr["K"][:, ::-1]
        func = sktrans.PolynomialTransform()
        if func.estimate(src=K_pts, dst=Q_pts, order=self.order):
            return func
        else:
            warnings.warn(
                "unable to fit polynomial transform of order 2", RuntimeWarning
            )
            return lambda x: x


ALIGNER_MAP = {
    "kabsch": KabschMapping,
    "polynomial2": lambda: PolynomialMapping(order=2),
    "polynomial3": lambda: PolynomialMapping(order=3),
    "polynomial4": lambda: PolynomialMapping(order=4),
}


def get_alignment_function(Q, K, corr, method_name="kabsch"):
    if corr["size"] < 3:
        warnings.warn("alignment is too weak", RuntimeWarning)
        return DummyMapping()
    elif method_name not in ALIGNER_MAP:
        warnings.warn("alignment is too weak", RuntimeWarning)
        return DummyMapping()
    return ALIGNER_MAP[method_name]()


def get_QK_correspondence(
    Q: ImageDesc, K: ImageDesc, extractor_name: str = "KAZE", epsilon: float = 0.01
):
    extractor = EXTRACTOR_MAP[extractor_name]()
    epsilon = max(0, epsilon)
    Q.points = extractor(Q.img)
    K.points = extractor(K.img)

    matcher = CliqueMatcher(epsilon=epsilon, use_dfs=False)
    return matcher(Q, K)


if __name__ == "__main__":
    main()
