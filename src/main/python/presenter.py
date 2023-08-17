import matplotlib
import skimage.io as skio


def draw_kde(ax, loader, etor, aligner, metric, score):
    fname = "{}-{}-{}".format(etor, aligner, metric)
    ax.set_title("{} score: {}".format(metric, score))


def write_plot(fig, sinfo):
    print(sinfo)
    gs = fig.add_gridspec(4, 4)

    q = sinfo["q"]
    k = sinfo["k"]
    corr = sinfo["corr"]

    if q.img.shape[0] < q.img.shape[1]:
        qp = fig.add_subplot(gs[0:2, 0:2])
        kp = fig.add_subplot(gs[2:, 0:2])
        urh = fig.add_subplot(gs[1:, 2:])
        logo = fig.add_subplot(gs[0:1, 2:])
    else:
        qp = fig.add_subplot(gs[0:, 0])
        kp = fig.add_subplot(gs[0:, 1])
        urh = fig.add_subplot(gs[1:, 2:])
        logo = fig.add_subplot(gs[0:1, 2:])

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
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
