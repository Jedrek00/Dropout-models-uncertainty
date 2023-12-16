from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.image as mpimg


def plot_history(history: dict, filepath: Optional[str] = None):
    x = range(len(history["train_loss"]))
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.plot(x, history["train_loss"], label="train_loss")
    plt.plot(x, history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(212)
    plt.plot(x, history["train_acc"], label="train_acc")
    plt.plot(x, history["val_acc"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    if filepath is not None:
        plt.savefig(filepath)


def plot_confusion_matrix(
    preds: list, labels: list, class_names: list, filepath: Optional[str] = None
):
    plt.figure(figsize=(12, 12))
    matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    disp.plot(xticks_rotation="vertical")
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath)


def plot_uncertainty(
    probs: list,
    labels: list,
    max_n: Optional[int] = None,
    separate: bool = True,
    img_path: Optional[str] = None,
    img_size: int = 32,
    filepath: Optional[str] = None,
):
    """
    Generate plot of model uncertainty based on probabilities given as a first argument.

    :param probs: probabilities array of each option get from softmax layer
    :param labels: list of labels
    :param max_n: defines how many best probabilities should be shown
    :param separate: if true, each label should be in separate column
    :param img_path: path to image that should be shown at the bottom of a plot
    :param img_size: size of image that is shown at the bottom of a plot
    :param filepath: path where plot should be saved as an image
    """
    POS_I = 1
    n = len(labels) if separate else 1
    max_n = max_n if max_n else n
    best_n = list(range(n))

    if max_n != None:
        sums = np.argsort(np.sum(probs.T, axis=1))[::-1]
        best_n = sums[:max_n]

    x = []
    legend_handles = []
    size = 50000 / max_n - 1000 * (max_n - 1) if separate else 50000

    if separate:
        x = [[i] * probs.shape[0] for i in range(len(best_n))]
    else:
        x = [[POS_I] * probs.shape[0] for _ in range(len(best_n))]

    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_n))

    trans_probs = probs.T[best_n, :]
    for x_p, val, p, i in zip(x, best_n, trans_probs, list(range(max_n))):
        scatter = plt.scatter(
            x_p, p, color=colors[i], s=size, marker="_", alpha=0.1, label=labels[val]
        )
        legend_handles.append(scatter)

    plt.ylim(0, 1)
    if separate:
        plt.xticks(list(range(max_n)), best_n)
    else:
        plt.xticks([POS_I], "")

    if img_path != None:
        img = plt.imread(img_path)
        ax = plt.gca()
        tick_labels = ax.xaxis.get_ticklabels()

        for tick in tick_labels:
            x_pos = tick.get_position() if separate else tick_labels[0].get_position()
            ib = OffsetImage(img, zoom=img_size / img.shape[0])
            ib.image.axes = ax
            ab = AnnotationBbox(ib, x_pos, frameon=False, box_alignment=(0.5, 1.2))
            ax.add_artist(ab)

    plt.ylabel("probability")
    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
        handle.set_alpha(1)

    if filepath != None:
        plt.savefig(filepath)
    plt.show()


def create_probs(n: int):
    probs = []
    for _ in range(n):
        rand_probs = np.random.rand(10)
        norm_probs = rand_probs / np.sum(rand_probs)
        probs.append(norm_probs)
    return np.array(probs)
