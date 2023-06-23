import pandas as pd
import glob
import sqlalchemy as sa


def save_meta():
    df = pd.read_csv("./inputs2_close_names.csv")

    def func1(x):
        return x.split("/")[0]

    def func2(x):
        x1 = x.split("/")[1].replace(".tiff", "")
        return x1

    df["Q_size"] = df["Q"].apply(func1)
    df["K_size"] = df["K"].apply(func1)

    df["Q"] = df["Q"].apply(func2)
    df["K"] = df["K"].apply(func2)

    def shoe(x):
        a = x.split("_")[:2]
        return "_".join(a)

    df["Q_shoe"] = df["Q"].apply(shoe)
    df["K_shoe"] = df["K"].apply(shoe)

    def repl(x):
        return x.split("_")[-1]

    df["Q_repl"] = df["Q"].apply(repl)
    df["K_repl"] = df["K"].apply(repl)

    df["pair"] = df.apply(lambda r: r["Q_shoe"] + "-" + r["K_shoe"], axis=1)
    df["size"] = df["Q_size"]

    sdf = df.loc[:, ("cmpid", "pair", "Q_shoe", "K_shoe", "Q", "K", "size", "flip_k")]
    print(sdf.head())

    sdf.to_csv("./inputs2_meta.csv", header=True, index=False)

    pass


def bind_everything():
    meta = pd.read_csv("./inputs2_meta.csv")

    df = pd.concat(map(pd.read_csv, glob.glob("./results2_e_*.csv")))
    df.drop(["corresponder", "eps2"], axis=1, inplace=True)

    full = df.join(meta, on="cmpid", lsuffix="", rsuffix="_r")
    full["epsilon"] = full["eps1"]
    full.drop(["cmpid_r", "eps1"], axis=1, inplace=True)

    shoe_columns = [
        "cmpid",
        "is_match",
        "close",
        "pair",
        "Q_shoe",
        "K_shoe",
        "Q",
        "K",
        "size",
        "flip_k",
    ]

    score_columns = [
        "blur",
        "extractor",
        "alignment",
        "metric",
        "epsilon",
        "alpha",
        "score",
    ]

    full = full[shoe_columns + score_columns]
    full["score"].round(6)
    print(full.head())
    print(len(full))

    # full.to_csv("./results2_FULL_SCORES.csv", header=True, index=False)

    engine = sa.create_engine("sqlite:///results2_FULL_SCORES.db")
    full.to_sql(
        name="results", con=engine, if_exists="replace", index=False, chunksize=100000
    )


def main():
    save_meta()
    bind_everything()


if __name__ == "__main__":
    main()
