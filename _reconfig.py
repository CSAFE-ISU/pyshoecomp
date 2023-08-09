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

valid_keys = ("ESY", "SC1")


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
        "CENSURE": dict(),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.2, min_distance=10),
        "KAZE": dict(threshold=0.03),
        "AKAZE": dict(threshold=0.03),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @staticmethod
    def get_params(name):
        return ESYConfig.internal[name]


class SC1Config:
    internal = {
        # related to reading in Q images
        "img_Q0": dict(scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False),
        "img_Q1": dict(scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False),
        # related to reading in K images
        "img_K0": dict(scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=True),
        "img_K1": dict(scale=0.25, crop=((80, 40), (0, 80)), x_flip=False, y_flip=False),
        # related to interest points
        "SIFT": dict(upsampling=1, sigma_in=0),
        "ORB": dict(fast_threshold=0.075),
        "CENSURE": dict(),
        "Shi-Tomasi": dict(maxCorners=500, qualityLevel=0.2, min_distance=10),
        "KAZE": dict(threshold=0.03),
        "AKAZE": dict(threshold=0.03),
        "FAST_peaks": dict(min_distance=7),
        "FAST_params": dict(threshold=0.075),
    }

    @staticmethod
    def get_params(name):
        return SC1Config.internal[name]


class Config:
    current = "ESY"
    mappings = {"ESY": ESYConfig, "SC1": SC1Config}

    @staticmethod
    def get_params(name):
        return Config.mappings[Config.current].get_params(name)
