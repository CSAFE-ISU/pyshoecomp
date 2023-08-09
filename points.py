import pandas as pd
import numpy as np
import time
import gc
import sys, os, argparse
import itertools

import matplotlib

matplotlib.use("pgf")
from matplotlib import rcParams, rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import skimage.transform as sktrans

plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\newfontfamily\Raleway[Ligatures=TeX]{Raleway}",
                r"\setsansfont{Raleway}[UprightFont=*-Regular,ItalicFont=*-Italic,BoldFont=*-Bold,BoldItalicFont=*-BoldItalic]",
                r"\setmainfont{Raleway}[UprightFont=*-Regular,ItalicFont=*-Italic,BoldFont=*-Bold,BoldItalicFont=*-BoldItalic]",
                r"\renewcommand{\seriesdefault}{\bfdefault}",
            ]
        ),
    }
)

RED = "#dc3220"
BLUE = "#005ab5"

from _reconfig import Config, valid_keys
from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP
from corresponder import CORRESPONDER_MAP
from aligner import (
    ALIGNER_MAP,
    get_QK_correspondence,
    get_alignment_function,
)
from scorer import SCORINGMETHOD_MAP

SCORES_PER_COMPARISON = (
    len(EXTRACTOR_MAP)
    * len(CORRESPONDER_MAP)
    * len(ALIGNER_MAP)
    * len(SCORINGMETHOD_MAP)
)

SUBMAP = {k: v for k, v in EXTRACTOR_MAP.items() if "+" not in k}


def runner(path):
    k = ImageDesc.from_file(path, is_k=True, is_match=True)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 9))
    fig.suptitle("What kinds of interest points?")
    axs = axs.ravel()

    zz = k.img
    axs[0].imshow(k.img, cmap="gray")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    for i, (k, v) in enumerate(SUBMAP.items()):
        axs[i + 1].set_xticks([])
        axs[i + 1].set_yticks([])

    fig.tight_layout()
    fig.savefig(f"plots/pts-0.pdf", format="pdf")

    for i, (k, v) in enumerate(SUBMAP.items()):
        axs[i + 1].imshow(zz, cmap="gray")
        etor = v()
        pts = etor(zz)
        axs[i + 1].scatter(x=pts[:, 1], y=pts[:, 0], c=BLUE, marker="o", s=5, alpha=1)
        axs[i + 1].set_title(k)
    fig.savefig(f"plots/pts-1.pdf", format="pdf")


def main():
    parser = argparse.ArgumentParser("view-points-pdf")
    parser.add_argument(
        "-i",
        "--path",
        dest="path",
        type=str,
        help="path of image",
        default="/home/gautham/stuff/CSAFE/Shoeprints/presentations/2023-aug-23-IAI/images/LOL.tiff",
    )
    parser.add_argument(
        "-x", "--config", default="ESY", help="config to use: " + str(valid_keys)
    )
    d = parser.parse_args()
    Config.current = d.config
    runner(d.path)


if __name__ == "__main__":
    main()
