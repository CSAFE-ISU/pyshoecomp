import joblib
import sys
import os
import pandas as pd
import time
import argparse
import numpy as np
import subprocess

EXECUTABLE_PATH = sys.executable

#
from _reconfig import Config, valid_keys

#


def check_csv(x):
    if (
        isinstance(x, str)
        and x.endswith(".csv")
        and os.path.exists(x)
        and os.path.isfile(x)
    ):
        df = pd.read_csv(x, header=0)
        assert (
            "Q" in df.columns
        ), "input csv must have a column Q with strings (file locations)"
        assert (
            "K" in df.columns
        ), "input csv must have a column K with strings (file locations)"
        assert (
            "match" in df.columns
        ), "input csv must have a column match with boolean values"
        assert "config" in df.columns, "input csv must have a column config"
        assert (
            "blur" in df.columns
        ), "input csv must have a column blur with strings 02 to 12"
        df["cmpid"] = np.arange(1, len(df) + 1)
        sdf = df[["cmpid", "Q", "K", "match", "config", "blur"]]
        sdf.to_csv(x.replace(".csv", "_names.csv"), index=False, header=True)
        return sdf.to_dict("records")
    else:
        raise RuntimeError(f"invalid file {x} provided")


def singleton(
    cmpid,
    q_path,
    k_path,
    is_match,
    config,
    blur,
    eps1,
    eps2,
    alpha,
    exclude,
    output,
):
    try:
        proc = subprocess.Popen(
            [
                EXECUTABLE_PATH,
                "./singleton.py",
                "--id",
                f"{cmpid}",
                "--q-path",
                f"{q_path}",
                "--k-path",
                f"{k_path}",
                "--match" if is_match else "--nonmatch",
                "--config",
                str(config),
                "--blur",
                str(blur),
                "--eps1",
                str(eps1),
                "--eps2",
                str(eps2),
                "--alpha",
                str(alpha),
                "--exclude",
                " ".join(exclude),
                "--output",
                str(output),
            ],
            cwd="./",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        out, err = proc.communicate()
        if "<<<SUCCEEDED>>>" in out:
            print("SUCCESS", out)
            return True
        else:
            print("ERROR", err)
            return False
    except Exception as e:
        print(e, "failed at input")
        return False


def output_header(output):
    column_names = [
        "cmpid",
        "config",
        "is_match",
        "blur",
        "extractor",
        "q_pts",
        "k_pts",
        "corresponder",
        "alignment",
        "metric",
        "score",
        "eps1",
        "alpha",
    ]
    df = pd.DataFrame([], columns=column_names)
    df.to_csv(output, header=True, index=False, mode="w")


def runner(filelist, eps1, eps2, alpha, exclude, output):
    output_header(output)
    results = []
    with joblib.Parallel(n_jobs=6, backend="loky") as parallel:
        results = parallel(
            joblib.delayed(singleton)(
                cmpid=x["cmpid"],
                q_path=x["Q"],
                k_path=x["K"],
                is_match=x["match"],
                config=x["config"],
                blur=x["blur"],
                eps1=eps1,
                eps2=eps2,
                alpha=alpha,
                exclude=exclude,
                output=output,
            )
            for x in filelist
        )
    successes = sum(results, 0)
    print(len(results), "successes = ", 100 * successes / len(results), "%")


def main():
    parser = argparse.ArgumentParser("compare-csv")
    parser.add_argument(
        "-i", "--infile", type=check_csv, help="input CSV containing paths of pairs"
    )
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
        "--exclude",
        required=False,
        default=[],
        nargs="*",
        help="cases to exclude(all inclusive)",
    )
    parser.add_argument(
        "-o", "--output", default="./results.csv", help="file location of output CSV"
    )
    d = parser.parse_args()
    result = runner(
        filelist=d.infile,
        eps1=d.eps1,
        eps2=d.eps2,
        alpha=d.alpha,
        exclude=set(d.exclude),
        output=d.output,
    )


if __name__ == "__main__":
    main()
