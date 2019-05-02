import matplotlib.pyplot as plt


def save_plot(x, y, path, x_label="", y_label=""):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y, 'b')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.savefig(path)
    plt.close(fig)
