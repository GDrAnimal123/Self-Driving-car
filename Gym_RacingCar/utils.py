import matplotlib.pyplot as plt
import numpy as np


def save_plot(x, y, path, x_label="", y_label=""):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y, 'b')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.savefig(path)
    plt.close(fig)


def standardize(x):
    if np.std(x):
        return (x - np.mean(x)) / np.std(x)
    else:
        print ('std = 0!')
        return 0


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
