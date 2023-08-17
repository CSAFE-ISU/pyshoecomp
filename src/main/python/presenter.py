import matplotlib


def write_plot(fig, sinfo):
    axes = fig.add_subplot(111)
    axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
