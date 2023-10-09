"""
the idea here is that different datasets might require
different processing parameters, so we should initialize
all those parameters using configurations here.

could this be a JSON file? probably, but I don't want to worry about
read/write/validate and all that stuff, I just want a config.

This is something like a #define that I would like to add in at runtime.

#ifdef ESY
SIFT_params = blah, blah
#endif

"""

__all__ = ("Config", "valid_keys")

valid_keys = ("ESY", "SC1", "SC2", "WV1", "WV2", "Dummy")


class DummyConfig:
    internal = {
        # related to reading in Q images
        "img_Q0": dict(scale=1, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_Q1": dict(scale=1, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to reading in  imges
        "img_K0": dict(scale=1, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_K1": dict(scale=1, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(non_max_threshold=0.1),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.2, minDistance=7),
        "KAZE": dict(threshold=0.03),
        "AKAZE": dict(threshold=0.03),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]


class ESYConfig:
    internal = {
        # related to reading in Q images
        "img_Q0": dict(
            scale=0.125, crop=((20, 20), (20, 20)), x_flip=False, y_flip=False
        ),
        "img_Q1": dict(
            scale=0.125, crop=((20, 20), (20, 20)), x_flip=False, y_flip=False
        ),
        # related to reading in K images
        "img_K0": dict(
            scale=0.125, crop=((20, 20), (20, 20)), x_flip=True, y_flip=False
        ),
        "img_K1": dict(
            scale=0.125, crop=((20, 20), (20, 20)), x_flip=False, y_flip=False
        ),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(non_max_threshold=0.1),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.2, minDistance=7),
        "KAZE": dict(threshold=0.03),
        "AKAZE": dict(threshold=0.03),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]


class SC1Config:
    internal = {
        # related to reading in Q images
        "img_Q0": dict(
            scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False
        ),
        "img_Q1": dict(
            scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False
        ),
        # related to reading in K images
        "img_K0": dict(scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=True),
        "img_K1": dict(
            scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False
        ),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(non_max_threshold=0.05),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.1, minDistance=7),
        "KAZE": dict(threshold=0.001),
        "AKAZE": dict(threshold=0.001),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]


class WV2Config:
    # Q is tile, K is handiprint
    r = 1.135
    r2 = 0.25
    internal = {
        # related to reading in Q images
        "img_Q0": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_Q1": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to reading in K images
        "img_K0": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_K1": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=True, y_flip=False),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(non_max_threshold=0.05),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.1, minDistance=7),
        "KAZE": dict(threshold=0.001),
        "AKAZE": dict(threshold=0.001),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.3),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]



class SC2Config:
    # Q is tile, K is handiprint
    r = 1.135
    r2 = 0.125
    internal = {
        # related to reading in Q images
        "img_Q0": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_Q1": dict(scale=r2, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to reading in K images
        "img_K0": dict(scale=r2/r, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_K1": dict(scale=r2/r, crop=((0, 0), (0, 0)), x_flip=True, y_flip=False),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(non_max_threshold=0.05),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.1, minDistance=7),
        "KAZE": dict(threshold=0.001),
        "AKAZE": dict(threshold=0.001),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]


class WV1Config:
    internal = {
        # related to reading in Q images
        "img_Q0": dict(scale=0.125, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        "img_Q1": dict(scale=0.125, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to reading in K images
        "img_K0": dict(scale=0.125, crop=((0, 0), (0, 0)), x_flip=False, y_flip=True),
        "img_K1": dict(scale=0.125, crop=((0, 0), (0, 0)), x_flip=False, y_flip=False),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.1),
        "CENSURE": dict(non_max_threshold=0.05),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.1, minDistance=7),
        "KAZE": dict(threshold=0.007),
        "AKAZE": dict(threshold=0.007),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.3),
    }

    @classmethod
    def get_params(cls, name):
        return cls.internal[name]


class Config:
    current = "ESY"
    mappings = {
        "ESY": ESYConfig,
        "SC1": SC1Config,
        "SC2": SC2Config,
        "WV1": WV1Config,
        "WV2": WV2Config,
        "Dummy": DummyConfig,
    }

    @staticmethod
    def get_params(name):
        return Config.mappings[Config.current].get_params(name)
