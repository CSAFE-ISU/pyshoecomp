#!/usr/bin/env python
# create the sample pairs to compare ShoeCase data (blood prints)
import sys
import glob
import os
import random
import pandas as pd
import argparse


def renumber(x):
    if x % 4 < 2:
        return x+2
    else:
        return x-2

def driver(target="./inputs.csv"):
    df = pd.read_csv("./datasets/names_SC1.txt", header=0)

    # matches
    match = pd.DataFrame(dict(Q=df["Q"], K=df["K"], match=True))

    # close nonmatches are flipped
    ind = list(map(renumber, range(len(df))))
    x = df["K"][ind].tolist()
    nonmatch = pd.DataFrame(dict(Q=df["Q"], K=x, match=False))

    res = pd.concat([match, nonmatch])
    res["config"] = "SC1"
    res["blur"] = df["K"].apply(lambda x: "20" if "FT" in x else "22")
    res.to_csv(target, header=True, index=False)
    print(len(res), "samples saved in", target)



def main():
    parser = argparse.ArgumentParser(prog="shoedata-sampler")
    parser.add_argument(
        "-o",
        "--output",
        default="./full_SC1.csv",
        help="target csv filename to save samples",
    )

    d = parser.parse_args()
    driver(target=d.output)


if __name__ == "__main__":
    main()
