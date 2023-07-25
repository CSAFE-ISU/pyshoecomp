from imdesc import ImageDesc
from extractor import EXTRACTOR_MAP
from corresponder import CORRESPONDER_MAP
from aligner import (
    ALIGNER_MAP,
    get_QK_correspondence,
    get_alignment_function,
)
from scorer import SCORINGMETHOD_MAP

import pandas as pd
import time
import gc
import sys, os, argparse
import itertools
import matplotlib.pyplot as plt
import skimage.transform as sktrans

SCORES_PER_COMPARISON = (
    len(EXTRACTOR_MAP)
    * len(CORRESPONDER_MAP)
    * len(ALIGNER_MAP)
    * len(SCORINGMETHOD_MAP)
)


def etor_check(x):
    if x in EXTRACTOR_MAP:
        return x
    raise RuntimeError(f"invalid aligner {x}")


def aligner_check(x):
    if x in ALIGNER_MAP:
        return x
    raise RuntimeError(f"invalid aligner {x}")


def argwrapper(func):
    def wfunc(*args, **kwargs):
        print(args)
        print(kwargs)
        # func(*args, **kwargs)

    return wfunc


def show_text(ax, base):
    ax.set_title("scores")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    for i, x in enumerate(base["scores"]):
        txt = "{} = {:.4f}".format(x["metric"], x["score"])
        ax.text(50, (i + 1) * 50, txt)
        # print(txt)


def show_overall(cmpid, base, save):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharex=True, sharey=True)
    Q = base["q"]
    K = base["k"]
    aligner = base["alignment"]
    corr = base["corr"]
    map_func = base["map_func"]
    is_match = base["is_match"]
    for i, desc in enumerate([Q, K]):
        axs[i].imshow(desc.img, cmap="gray")
        continue
        axs[i].scatter(
            x=desc.points[:, 1],
            y=desc.points[:, 0],
            c="red",
            s=5,
            marker="x",
            alpha=1,
        )
    axs[0].scatter(
        x=corr["Q"][:, 1], y=corr["Q"][:, 0], c="red", marker="o", s=5, alpha=1
    )
    axs[1].scatter(
        x=corr["K"][:, 1], y=corr["K"][:, 0], c="red", marker="o", s=5, alpha=1
    )
    axs[0].set_title("Q")
    axs[1].set_title("K")
    title = "cmp_{} ({}): align {} via {}".format(
        cmpid, "matched" if is_match else "nonmatch", base["extractor"], aligner
    )
    show_text(axs[2], base)
    fig.suptitle(title)
    fig.tight_layout()

    if save:
        fig.savefig(f"cmp_{cmpid}.pdf", format="pdf")
    else:
        plt.show()


def runner(
    cmpid,
    k_path,
    q_path,
    is_match,
    flip_k,
    etor,
    aligner,
    epsilon1,
    epsilon2,
    alpha,
    save,
):
    k = ImageDesc.from_file(
        k_path,
        scale_factor=0.125,
        outer_crop=20,
        flip=flip_k,
    )
    q = ImageDesc.from_file(
        q_path,
        scale_factor=0.125,
        outer_crop=20,
    )
    result = []
    scor = None

    extractor = EXTRACTOR_MAP[etor]()
    q.points = extractor(q.img)
    k.points = extractor(k.img)

    gc.collect()
    cder = CORRESPONDER_MAP["clique2"](epsilon=epsilon1, epsilon2=epsilon2, alpha=alpha)
    corr = cder(q, k)

    mapping = get_alignment_function(q, k, corr, method_name=aligner)
    map_func = mapping(q, k, corr)
    q.aligned_img = mapping.align_Q_to_K(q, k, corr, map_func=map_func)

    base = {
        "q": q,
        "k": k,
        "corr": corr,
        "map_func": map_func,
        "is_match": is_match,
        "extractor": etor,
        "q_pts": len(q.points),
        "k_pts": len(k.points),
        "corresponder": "clique2",
        "alignment": aligner,
        "eps1": epsilon1,
        "eps2": epsilon2,
        "alpha": alpha,
        "scores": [],
    }

    for s in SCORINGMETHOD_MAP.keys():
        scor = SCORINGMETHOD_MAP[s](
            Q=q, K=k, corr=corr, map_func=map_func, epsilon=epsilon2
        )
        try:
            point = scor()
            entry = {
                "metric": s,
                "score": point,
            }
            result.append(entry)
        except Exception as e:
            print(e, "failure with", s)

    base["scores"] = result
    # print(base)
    show_overall(cmpid=cmpid, base=base, save=save)


def main():
    parser = argparse.ArgumentParser("view-single")
    parser.add_argument(
        "-i", "--id", dest="_id", required=True, help="ID of comparison"
    )
    parser.add_argument(
        "-k", "--k-path", dest="k_path", type=str, required=True, help="path of K"
    )
    parser.add_argument(
        "-q", "--q-path", dest="q_path", type=str, required=True, help="path of Q"
    )
    # is it a match?
    parser.add_argument("--match", dest="is_match", action="store_true")
    parser.add_argument("--nonmatch", dest="is_match", action="store_false")
    # do I have to flip K?
    parser.add_argument("--flip-k", dest="flip_k", action="store_true")
    parser.add_argument("--no-flip-k", dest="flip_k", action="store_false")
    # eps
    parser.add_argument(
        "--eps1",
        type=float,
        default=0.5,
        help="epsilon tolerance value for graph construction",
    )
    # the median distance param
    parser.add_argument(
        "--eps2",
        type=float,
        default=5,
        help="epsilon tolerance value for graph similarity",
    )
    # alpha distance for clique search
    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="tolerance value for removing neighbor points",
    )
    #
    parser.add_argument(
        "-e", "--etor", default="AKAZE", help="type of extractor", type=etor_check
    )
    parser.add_argument(
        "-a", "--aligner", default="kabsch", help="type of aligner", type=aligner_check
    )
    parser.add_argument("--save", dest="should_save", action="store_true")
    parser.set_defaults(is_match=True, flip_k=False, should_save=False)
    d = parser.parse_args()
    result = runner(
        cmpid=d._id,
        k_path=d.k_path,
        q_path=d.q_path,
        is_match=d.is_match,
        flip_k=d.flip_k,
        epsilon1=d.eps1,
        epsilon2=d.eps2,
        alpha=d.alpha,
        etor=d.etor,
        aligner=d.aligner,
        save=d.should_save,
    )


if __name__ == "__main__":
    main()
