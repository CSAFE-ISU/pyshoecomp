#!/usr/bin/env python
# create the sample pairs to compare
# by randomly sampling from the list of available shoes
import sys
import glob
import os
import random
import pandas as pd
import argparse

def deconstruct(name):
    folder, raw = name.split("/")
    img, ext = raw.split(".")
    ID, foot, blur, replicate = img.split("_")

    return dict(
        folder=folder, ID=ID, foot=foot, blur=blur, replicate=replicate, ext=ext
    )


def reconstruct(data):
    return "{}/{}_{}_{}_{}.{}".format(
        data["folder"],
        data["ID"],
        data["foot"],
        data["blur"],
        data["replicate"],
        data["ext"],
    )


def get_sample_list(num_samples, Qnames, Knames, match=True, flip_close=True):
    answer = []
    for i in range(num_samples):
        Q = random.sample(Qnames, 1)[0]
        Qdat = deconstruct(Q)
        Kdat = dict(**Qdat)
        if match:
            # match, so K should be the 'non-blurred' version
            # of the same shoe
            Kdat["blur"] = "00"
        elif flip_close:
            # close-nonmatch, but flipped, so K should
            # be the 'non-blurred' version of the opposite shoe
            Kdat["blur"] = "00"
            Kdat["foot"] = "L" if Qdat["foot"] == "R" else "R"
        else:
            # do far-nonmatches make sense?
            while True:
                Kdat = deconstruct(random.sample(Knames, 1)[0])
                if Kdat["folder"] == Qdat["folder"] and Kdat["ID"] != Qdat["ID"]:
                    break
        Kdat["replicate"] = str(random.randint(1, 3))

        row = {
            "Q": reconstruct(Qdat),
            "K": reconstruct(Kdat),
            "match": match,
            "close": flip_close,
            "flip_k": Qdat["foot"] != Kdat["foot"],
            "blur": Qdat["blur"],
        }
        answer.append(row)
    return answer


def driver(num_samples=100, use_farnon=True, target="./inputs.csv"):
    base = open("./names.txt").readlines()
    subtypes = ["K", "02", "04", "06", "08", "10", "12"]
    dset = {x: [] for x in subtypes}
    for name in map(lambda x: x.strip(), base):
        if "00_" in name:
            dset["K"].append(name)
        else:
            sub = deconstruct(name)["blur"]
            dset[sub].append(name)

    res = []
    for x in subtypes[1:]:
        res += get_sample_list(
            num_samples, dset[x], dset["K"], match=True, flip_close=False
        )
        res += get_sample_list(
            num_samples, dset[x], dset["K"], match=False, flip_close=True
        )
        if use_farnon:
            res += get_sample_list(
                num_samples, dset[x], dset["K"], match=False, flip_close=False
            )

    df = pd.DataFrame(res)
    df.to_csv(target, header=True, index=False)
    print(len(df), "samples saved in", target)


def main():
    parser = argparse.ArgumentParser(prog="shoedata-sampler")
    parser.add_argument(
        "-n",
        "--num-samples",
        default=100,
        type=int,
        help="number of samples per category",
    )
    parser.add_argument(
        "-z",
        "--use-far-nonmatches",
        default=False,
        dest="use_farnon",
        help="if False, non matches are just the flipped images of the opposite shoe",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./inputs.csv",
        help="target csv filename to save samples",
    )

    d = parser.parse_args()
    driver(num_samples=d.num_samples, use_farnon=d.use_farnon, target=d.output)


if __name__ == "__main__":
    main()
