import joblib
import sys
import os
import pandas as pd
import time
import argparse
import numpy as np
import subprocess

EXECUTABLE_PATH = sys.executable


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
        assert (
            "config" in df.columns
        ), "input csv must have a column config"
        assert (
            "blur" in df.columns
        ), "input csv must have a column blur with strings 02 to 12"
        df["cmpid"] = np.arange(1, len(df) + 1)
        sdf = df[["cmpid", "config", "Q", "K", "match", "blur"]]
        # sdf.to_csv(x.replace(".csv", "_names.csv"), index=False, header=True)
        return sdf.to_dict("records")
    else:
        raise RuntimeError(f"invalid file {x} provided")


def singleton(
    cmpid, q_path, k_path, is_match, config, eps1, eps2, alpha, etor, aligner, output
):
    try:
        proc = subprocess.Popen(
            [
                EXECUTABLE_PATH,
                "./view_compare.py",
                "--id",
                f"{cmpid}",
                "--q-path",
                f"{q_path}",
                "--k-path",
                f"{k_path}",
                "--match" if is_match else "--nonmatch",
                "--config",
                str(config),
                "--eps1",
                str(eps1),
                "--eps2",
                str(eps2),
                "--alpha",
                str(alpha),
                "--etor",
                str(etor),
                "--aligner",
                str(aligner),
                "--save",
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


def runner(filelist, eps1, eps2, alpha, etor, aligner, output):
    results = []
    with joblib.Parallel(n_jobs=4, backend="loky") as parallel:
        results = parallel(
            joblib.delayed(singleton)(
                cmpid=x["cmpid"],
                q_path=x["Q"],
                k_path=x["K"],
                is_match=x["match"],
                config=x["config"],
                eps1=eps1,
                eps2=eps2,
                alpha=alpha,
                etor=etor,
                aligner=aligner,
                output=output,
            )
            for x in filelist
        )
    successes = sum(results, 0)
    print(len(results), "successes = ", 100 * successes / len(results), "%")


def main():
    parser = argparse.ArgumentParser("compare-view-csv")
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
        "-e",
        "--etor",
        default="AKAZE",
        help="type of extractor",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--aligner",
        default="kabsch",
        help="type of aligner",
        type=str,
    )
    parser.add_argument("-o", "--output", default="./", help="output folder")
    d = parser.parse_args()
    result = runner(
        filelist=d.infile,
        eps1=d.eps1,
        eps2=d.eps2,
        alpha=d.alpha,
        etor=d.etor,
        aligner=d.aligner,
        output=d.output,
    )


if __name__ == "__main__":
    main()
