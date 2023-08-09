from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP
from neuralnet import avg_ncc, avg_poc

import skimage.feature as skfeat
import numpy as np

class ScoringMethod:
    _extname_ = "<unk>"

    def __init__(self, Q, K, corr, map_func, *args, **params):
        self.Q = Q
        self.K = K
        self.corr = corr
        self.map_func = map_func

    def __call__(self):
        raise NotImplementedError("base class")


class DummyTime(ScoringMethod):
    _extname_ = "time"

    def __call__(self):
        return self.corr["time"]


class CliqueSize(ScoringMethod):
    _extname_ = "clique_size"

    def __call__(self):
        if len(self.Q.points) <= 3:
            return 0
        # yeah we can probably redo the clique search
        # with just dx/dy, or
        # a condition with a smaller range of theta
        return self.corr["size"]


class CliqueFraction(ScoringMethod):
    _extname_ = "clique_fraction"

    def __call__(self):
        if len(self.Q.points) <= 3:
            return 0
        # return a ratio of clique size instead of an absolute value
        # saying "this fraction of points in Q got mapped to K"
        return self.corr["size"] / len(self.Q.points)


class OverlapPercentage(ScoringMethod):
    _extname_ = "overlap_percentage"

    def __init__(self, Q, K, corr, map_func, epsilon):
        super().__init__(Q, K, corr, map_func, epsilon)
        self.epsilon = epsilon

    def __call__(self):
        if len(self.Q.points) <= 10:
            return 0
        Q_pts = self.Q.points[:, ::-1]
        K_pts = self.K.points[:, ::-1]
        K_pts_in_Q_space = self.map_func(K_pts)
        indices = skfeat.match_descriptors(
            Q_pts, K_pts_in_Q_space, metric="euclidean", max_distance=self.epsilon
        )

        if len(indices) <= 10:
            return 0

        Q_close = Q_pts[indices[:, 0]]
        K_close = K_pts_in_Q_space[indices[:, 1]]
        dist = np.sqrt(np.sum((Q_close - K_close) ** 2, axis=1))

        overlap_count = np.sum(dist < self.epsilon)
        return overlap_count / len(Q_pts)


class MedianDistance(ScoringMethod):
    _extname_ = "median_distance"

    def __init__(self, Q, K, corr, map_func, epsilon):
        super().__init__(Q, K, corr, map_func, epsilon)
        self.epsilon = epsilon

    def __call__(self):
        if len(self.Q.points) <= 3:
            return 0
        Q_pts = self.Q.points[:, ::-1]
        K_pts = self.K.points[:, ::-1]
        K_pts_in_Q_space = self.map_func(K_pts)
        indices = skfeat.match_descriptors(
            Q_pts, K_pts_in_Q_space, metric="euclidean", max_distance=self.epsilon
        )

        Q_close = Q_pts[indices[:, 0]]
        K_close = K_pts_in_Q_space[indices[:, 1]]
        dist = np.sqrt(np.sum((Q_close - K_close) ** 2, axis=1))
        invdist = 1 / (1 + np.median(dist))
        return invdist


class NCC(ScoringMethod):
    _extname_ = "ImageNCC"

    @staticmethod
    def normalize(x):
        std = np.std(x)
        if std == 0:
            return 0
        return (x - np.mean(x)) / std

    def __call__(self):
        q_img = NCC.normalize(self.Q.aligned_img)
        k_img = NCC.normalize(self.K.img)
        return np.mean(q_img * k_img)


class POC_R(ScoringMethod):
    _extname_ = "ImagePOC"

    def __call__(self):
        freq_space = np.fft.fft2(self.Q.aligned_img) * np.conj(np.fft.fft2(self.K.img))
        freq_space = freq_space / np.abs(freq_space)
        score = np.fft.irfft2(freq_space)  # imaginary part is ZERO
        result = np.max(score)
        return result


class MCNCC(ScoringMethod):
    _extname_ = "MCNCC"

    def __call__(self):
        # IMPL
        return avg_ncc(self.Q.aligned_img, self.K.img)


class MCPOC(ScoringMethod):
    _extname_ = "MCPOC"

    def __call__(self):
        # IMPL
        return avg_poc(self.Q.aligned_img, self.K.img)


SCORINGMETHOD_MAP = {x._extname_: x for x in ScoringMethod.__subclasses__()}
