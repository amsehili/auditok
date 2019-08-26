import matplotlib.pyplot as plt
import numpy as np


def plot(data, sampling_rate, show=True):
    y = np.asarray(data)
    ymax = np.abs(y).max()
    nb_samples = y.shape[-1]
    sample_duration = 1 / sampling_rate
    x = np.linspace(0, sample_duration * (nb_samples - 1), nb_samples)
    plt.plot(x, y / ymax, c="#024959")
    plt.ylim(-3, 3)
    if show:
        plt.show()


def plot_detections(data, sampling_rate, detections, show=True, save_as=None):

    plot(data, sampling_rate, show=False)
    if detections is not None:
        for (start, end) in detections:
            plt.axvspan(start, end, facecolor="g", ec="r", lw=2, alpha=0.4)

    if save_as is not None:
        plt.savefig(save_as, dpi=120)

    if show:
        plt.show()
    return
