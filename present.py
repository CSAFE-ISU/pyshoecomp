"""
presentation related scripts,
using matplotlib as an import
"""

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

# local imports
from imdesc import ImageDesc
from extractor import Extractor
from aligner import AlignFunction


def etor_plot_points_on_axis(
    etor: Extractor, axis: Any, img: ArrayLike, points: ArrayLike
):
    if points is None:
        points = etor(img)
    axis.imshow(img, cmap="gray")
    # remember you get points in row / col format, so swap for x / y
    axis.scatter(x=points[:, 1], y=points[:, 0], c="red")
    axis.set_xticks([])
    axis.set_yticks([])
    return axis


def etor_show_points(etor: Extractor, img: ArrayLike, points: ArrayLike):
    fig, ax = plt.subplots()
    etor_plot_points_on_axis(etor, ax, img, points)
    plt.show()


def show_alignment(alg: AlignFunction, Q, K, corr, *args, **params):
    map_func = params.get("map_func", alg._get_mapping(Q, K, corr, *args, **params))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    for i, desc in enumerate([Q, K]):
        axs[i].imshow(desc.img, cmap="gray")
        axs[i].scatter(
            x=desc.points[:, 1],
            y=desc.points[:, 0],
            c="red",
            s=3,
            marker="x",
            alpha=0.5,
        )
    axs[0].scatter(
        x=corr["Q"][:, 1], y=corr["Q"][:, 0], c="green", marker="o", s=5, alpha=1
    )
    axs[1].scatter(
        x=corr["K"][:, 1], y=corr["K"][:, 0], c="green", marker="o", s=5, alpha=1
    )
    Q.aligned_img = alg.align_Q_to_K(Q, K, corr, map_func=map_func)
    axs[2].imshow(Q.aligned_img, cmap="gray")
    axs[2].imshow(K.img, cmap="Reds_r", alpha=0.3)

    axs[0].set_title("Q")
    axs[1].set_title("K")
    axs[2].set_title("overlay K on transformed Q")
    fig.suptitle(f"alignment via {alg._extname_}")
    plt.show()
