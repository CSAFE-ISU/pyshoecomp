import matplotlib
import skimage.io as skio
import numpy as np


def draw_kde(ax, loader, etor, aligner, metric, score):
    fname0 = "{}-{}-{}.npy".format(etor, aligner, metric)
    fname = loader(fname0)
    subd = np.load(fname, allow_pickle=True)  # ouch
    matches = subd[()]["matches"]
    nonmatches = subd[()]["nonmatches"]

    match_hist = np.histogram(matches, bins=35, density=False)
    nonmatch_hist = np.histogram(nonmatches, bins=35, density=False)

    # print(type(ax))
    w = (np.max(match_hist[1]) - np.min(match_hist[1])) / len(match_hist[0])
    # $print(match_hist, "\n w=", w)
    ax.bar(
        match_hist[1][:-1],
        match_hist[0],
        width=w,
        color="#FF000088",
        edgecolor="#FF0000FF",
        label="match",
    )

    ax.bar(
        nonmatch_hist[1][:-1],
        nonmatch_hist[0],
        width=w,
        color="#0000FF88",
        edgecolor="#0000FFFF",
        label="nonmatch",
    )

    ax.set_xlabel(metric)
    ax.set_ylabel("Number of Pairs")
    ax.legend()

    if score is not None:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        delx = xlim[1] - xlim[0]
        if score < xlim[1] - 0.3 * delx:
            tx = score + 0.1 * delx
        else:
            tx = score - 0.3 * delx
        ty = np.mean(ylim)

        ax.axvline(
            x=score, ymin=ylim[0], ymax=ylim[1], color="black", linestyle="dashed"
        )
        ax.annotate(
            "you are here\n(%.4f)" % (score),
            xy=(score, np.mean(ylim)),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

    ax.set_title("{} score: {}".format(metric, score))


def write_plot(fig, sinfo):
    # print(sinfo)
    gs = fig.add_gridspec(6, 4)

    q = sinfo["q"]
    k = sinfo["k"]
    corr = sinfo["corr"]

    if q.img.shape[0] < q.img.shape[1]:
        qp = fig.add_subplot(gs[0:3, 0:2])
        kp = fig.add_subplot(gs[3:, 0:2])
        urh = fig.add_subplot(gs[2:, 2:])
        logo = fig.add_subplot(gs[0:2, 2:])
    else:
        qp = fig.add_subplot(gs[0:, 0])
        kp = fig.add_subplot(gs[0:, 1])
        urh = fig.add_subplot(gs[2:, 2:])
        logo = fig.add_subplot(gs[0:2, 2:])

    lname = sinfo["loader"]("csafe-logo.png")
    limg = skio.imread(lname)
    logo.imshow(limg)
    logo.axis("off")

    qp.set_xticks([])
    qp.set_yticks([])
    kp.set_xticks([])
    kp.set_yticks([])
    logo.set_xticks([])
    logo.set_yticks([])

    qp.imshow(q.img, cmap="Greys_r")
    qp.scatter(
        x=q.points[:, 1],
        y=q.points[:, 0],
        c="yellow",
        s=3,
        marker="x",
        alpha=1,
    )
    qp.scatter(x=corr["Q"][:, 1], y=corr["Q"][:, 0], c="red", marker="o", s=5, alpha=1)
    qp.set_title("Q")

    kp.imshow(k.img, cmap="Greys_r")
    kp.scatter(
        x=k.points[:, 1],
        y=k.points[:, 0],
        c="yellow",
        s=3,
        marker="x",
        alpha=1,
    )
    kp.scatter(x=corr["K"][:, 1], y=corr["K"][:, 0], c="red", marker="o", s=5, alpha=1)
    kp.set_title("K")

    draw_kde(
        urh,
        sinfo["loader"],
        sinfo["extractor"],
        sinfo["alignment"],
        sinfo["metric"],
        sinfo["score"],
    )

    fig.suptitle("shoecomp example output")
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
