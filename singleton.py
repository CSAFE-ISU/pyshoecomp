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

SCORES_PER_COMPARISON = (
    len(EXTRACTOR_MAP)
    * len(CORRESPONDER_MAP)
    * len(ALIGNER_MAP)
    * len(SCORINGMETHOD_MAP)
)


def runner(
    cmpid,
    k_path,
    q_path,
    exclude,
    is_match=True,
    flip_k=False,
    close=False,
    blur="00",
    epsilon1=1.5,
    epsilon2=1.5,
    alpha=0.01,
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
    etor = None
    cder = None
    corr = None
    align = None
    scor = None
    previous = ["", "", "", ""]
    for e, c, a, s in itertools.product(
        EXTRACTOR_MAP.keys(),
        CORRESPONDER_MAP.keys(),
        ALIGNER_MAP.keys(),
        SCORINGMETHOD_MAP.keys(),
    ):
        if len(set([e, c, a, s]) & exclude) > 0:
            continue
        if previous[0] != e:
            previous = [e, "", "", ""]
            extractor = EXTRACTOR_MAP[e]()
            q.points = extractor(q.img)
            k.points = extractor(k.img)
        if previous[1] != c:
            gc.collect()
            previous = [e, c, "", ""]
            cder = CORRESPONDER_MAP[c](epsilon=epsilon1, epsilon2=epsilon2, alpha=alpha)
            corr = cder(q, k)
        if previous[2] != a:
            previous = [e, c, a, ""]
            mapping = get_alignment_function(q, k, corr, method_name=a)
            map_func = mapping(q, k, corr)
            q.aligned_img = mapping.align_Q_to_K(q, k, corr, map_func=map_func)
        if previous[3] != s:
            previous[3] = s
            scor = SCORINGMETHOD_MAP[s](
                Q=q, K=k, corr=corr, map_func=map_func, epsilon=epsilon2
            )
            try:
                point = scor()
                entry = {
                    "cmpid": cmpid,
                    "is_match": is_match,
                    "close": close,
                    "blur": blur,
                    "extractor": e,
                    "corresponder": c,
                    "alignment": a,
                    "metric": s,
                    "score": point,
                    "eps1": epsilon1,
                    "eps2": epsilon2,
                    "alpha": alpha,
                }
                result.append(entry)
            except Exception as e:
                print(e, "failure with", s)

        previous = [e, c, a, s]
    if len(result) == SCORES_PER_COMPARISON:
        print("<<<SUCCEEDED>>>")
    return result


def main():
    parser = argparse.ArgumentParser("cmp-single")
    parser.add_argument(
        "-i", "--id", dest="_id", required=True, help="ID of comparison"
    )
    parser.add_argument(
        "-k", "--k-path", dest="k_path", type=str, required=True, help="path of K"
    )
    parser.add_argument(
        "-q", "--q-path", dest="q_path", type=str, required=True, help="path of Q"
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="path of resulting csv",
    )
    # is it a match?
    parser.add_argument("--match", dest="is_match", action="store_true")
    parser.add_argument("--nonmatch", dest="is_match", action="store_false")
    # is it a close nonmatch?
    parser.add_argument("--close", dest="close", action="store_true")
    parser.add_argument("--not-close", dest="close", action="store_false")
    # do I have to flip K?
    parser.add_argument("--flip-k", dest="flip_k", action="store_true")
    parser.add_argument("--no-flip-k", dest="flip_k", action="store_false")
    # how much blur in Q?
    parser.add_argument("--blur", default="77", type=str, help="blur")
    # eps
    parser.add_argument(
        "--eps1",
        type=float,
        default=0.5,
        help="epsilon tolerance value for graph construction",
    )
    parser.add_argument(
        "--eps2",
        type=float,
        default=5,
        help="epsilon tolerance value for graph similarity",
    )
    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="tolerance value for removing neighbor points",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        required=False,
        default=[],
        nargs="*",
        help="cases to exclude(all inclusive)",
    )
    parser.set_defaults(is_match=True, flip_k=False)
    d = parser.parse_args()
    result = runner(
        cmpid=d._id,
        k_path=d.k_path,
        q_path=d.q_path,
        exclude=set(d.exclude),
        is_match=d.is_match,
        close=d.close,
        blur=d.blur,
        flip_k=d.flip_k,
        epsilon1=d.eps1,
        epsilon2=d.eps2,
        alpha=d.alpha,
    )

    df = pd.DataFrame(result)
    df.to_csv(d.output, mode="a", index=False, header=False)


if __name__ == "__main__":
    main()
