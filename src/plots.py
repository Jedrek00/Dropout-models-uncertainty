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


def plot_morph_uncertainty(
    probs: [list],
    probs_count: int,
    img_count: int,
    labels: list,
    img_dir: str,
    max_n: Optional[int] = None,
    img_size: int = 32,
    filepath: Optional[str] = None,
    plot_title: Optional[str] = None,
    params: Optional[dict] = None,
):
    """
    Plot uncertainty based on morphological created images.

    Args:
        probs (list): List of probabilities.
        probs_count (int): Number of probabilities.
        img_count (int): Number of images.
        labels (list): List of labels.
        img_dir (str): Directory containing the images.
        max_n (int, optional): Maximum number of images to plot. Defaults to None.
        img_size (int, optional): Size of the images. Defaults to 32.
        filepath (str, optional): Filepath to save the plot. Defaults to None.
        plot_title (str, optional): Title of the plot. Defaults to None.
    """
    n = len(labels)
    max_n = max_n if max_n else n
    best_n = list(range(n))
    images = img_dir.split("-morph-")

    if max_n != None:
        sums = np.argsort(np.sum(probs.T, axis=(1, 2)))[::-1]
        best_n = sums[:max_n]

    legend_handles = []
    size = 5000
    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_n))

    plt.figure(figsize=(20, 10))
    if filepath == None:
        if params != None:
            plt.title(f"{params['first_img']} into {params['second_img']} | model: {params['model']} | type: {params['dropout_type']} | rate: {params['dropout_rate']}")
        else:
            plt.title(f"{images[0]} into {images[1]}")
    else:
        plt.title(plot_title)
    plt.ylim(0, 1.1)
    plt.xticks(list(range(img_count)), [""] * img_count)
    plt.ylabel("probability")
    ax = plt.gca()
    tick_labels = ax.xaxis.get_ticklabels()

    for step in range(img_count):
        trans_probs = probs[step].T[best_n, :]
        for val, p, i in zip(best_n, trans_probs, list(range(max_n))):
            scatter = plt.scatter(
                [step] * probs_count,
                p,
                color=colors[i],
                s=size,
                marker="_",
                alpha=0.1,
                label=labels[val],
            )

            if step == 0:
                legend_handles.append(scatter)

        img = plt.imread(f"{img_dir}/{step}.png")
        x_pos = tick_labels[step].get_position()
        ib = OffsetImage(img, zoom=img_size / img.shape[0])
        ib.image.axes = ax
        ab = AnnotationBbox(ib, x_pos, frameon=False, box_alignment=(0.5, 1.2))
        ax.add_artist(ab)

    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
        handle.set_alpha(1)

    if filepath != None:
        plt.savefig(filepath)
    else:
        plt.show()


def create_probs(n: int):
    probs = []
    for _ in range(n):
        rand_probs = np.random.rand(10)
        norm_probs = rand_probs / np.sum(rand_probs)
        probs.append(norm_probs)
    return np.array(probs)
