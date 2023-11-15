import argparse
import json
import os
import skimage.io as skio

#
from _reconfig import Config, valid_keys
from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP


def runner(path, ip, out):
    if ip not in EXTRACTOR_MAP:
        raise RuntimeError("invalid extractor, use: " + str(EXTRACTOR_MAP.keys()))
    k = ImageDesc.from_file(path, is_k=True, is_match=True)
    extractor = EXTRACTOR_MAP[ip]()
    k.points = extractor(k.img)

    result = {
        "filename": path,
        "etor": ip,
        "valid_points": (k.points).tolist(),
        "invalid_points": [],
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    imgpath = os.path.splitext(out)[0] + ".png"
    skio.imsave(imgpath, k.img)


def main():
    parser = argparse.ArgumentParser("extract-points")
    parser.add_argument(
        "-i",
        "--path",
        dest="path",
        type=str,
        help="path of image",
        default="./sample.tiff",
    )
    parser.add_argument(
        "-x",
        "--extractor",
        default="FAST",
        help="type of interest point: " + str(EXTRACTOR_MAP.keys()),
    )
    parser.add_argument(
        "-o", "--output", default="./result.json", help="location of output JSON file"
    )
    Config.current = "ESY"
    d = parser.parse_args()
    runner(d.path, d.extractor, d.output)


if __name__ == "__main__":
    main()
