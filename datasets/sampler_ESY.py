#!/usr/bin/env python
# create the sample pairs to compare EverSpry data
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
            "config": "ESY",
            "blur": Qdat["blur"],
        }
        answer.append(row)
    return answer


def get_sample_list_nr(num_samples, Qnames, Knames, match=True, flip_close=True):
    answer = []
    for i in range(len(Qnames)):
        Q = Qnames[i]
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
            "config": "ESY",
            "blur": Qdat["blur"],
        }
        answer.append(row)
    return answer


def driver(target="./inputs.csv"):
    base = open("./datasets/names_ESY.txt").readlines()
    subtypes = ["K", "02", "04", "06", "08", "10", "12"]
    dset = {x: [] for x in subtypes}
    for name in map(lambda x: x.strip(), base):
        if "00_" in name:
            dset["K"].append(name)
        else:
            sub = deconstruct(name)["blur"]
            dset[sub].append(name)

    res = []
    for x in subtypes[:-1]:
        res += get_sample_list_nr(
            num_samples, dset[x], dset["K"], match=True, flip_close=False
        )
        res += get_sample_list_nr(
            num_samples, dset[x], dset["K"], match=False, flip_close=True
        )

    df = pd.DataFrame(res)
    df.to_csv(target, header=True, index=False)
    print(len(df), "samples saved in", target)


def main():
    parser = argparse.ArgumentParser(prog="shoedata-sampler")
    parser.add_argument(
        "-o",
        "--output",
        default="./full_ESY.csv",
        help="target csv filename to save samples",
    )

    d = parser.parse_args()
    driver(target=d.output)


if __name__ == "__main__":
    main()
