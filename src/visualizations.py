from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_history(history: dict, filepath: Optional[str] = None):
    """
    Plots training and validation loss, and accuracy from a history dictionary.

    :param history (dict): A dictionary containing training and validation metrics.
                           It should have keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    :param filepath (str, optional): If provided, the plot will be saved to this file path.
    :return: None
    """
    x = range(len(history["train_loss"]))
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

    axes[0].plot(x, history["train_loss"], label="train_loss")
    axes[0].plot(x, history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(x, history["train_acc"], label="train_acc")
    axes[1].plot(x, history["val_acc"], label="val_acc")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)


def plot_confusion_matrix(
    preds: list, labels: list, class_names: list, filepath: Optional[str] = None
):
    """
    Plots the confusion matrix adn save it to file.

    :param preds (list): Predicted labels.
    :param labels (list): True labels.
    :param class_names (list): List of class names.
    :param filepath (str, optional): If provided, the plot will be saved to this file path.
    :return: None
    """
    matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation="vertical")
    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath)


def plot_uncertainty(
    probs: list,
    labels: list,
    max_display: Optional[int] = None,
    separate_columns: bool = True,
    img_path: Optional[str] = None,
    img_size: int = 32,
    save_path: Optional[str] = None,
):
    """
    Generate plot of model uncertainty based on probabilities given as a first argument.

    :param probs: probabilities array of each option get from softmax layer
    :param labels: list of labels
    :param max_n: defines how many best probabilities should be shown
    :param separate_columns: if true, each label should be in separate column
    :param img_path: path to image that should be shown at the bottom of a plot
    :param img_size: size of image that is shown at the bottom of a plot
    :param save_path: path where plot should be saved as an image
    """
    num_labels = len(labels)
    num_display = num_labels if separate_columns else 1
    max_display = max_display if max_display else num_display
    best_indices = list(range(num_display))

    if max_display is not None:
        sums = np.argsort(np.sum(probs.T, axis=1))[::-1]
        best_indices = sums[:max_display]

    x_values = []
    legend_handles = []
    size = 50000 / max_display - 1000 * (max_display - 1) if separate_columns else 50000

    if separate_columns:
        x_values = [[i] * probs.shape[0] for i in range(len(best_indices))]
    else:
        x_values = [[1] * probs.shape[0] for _ in range(len(best_indices))]

    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_display))

    trans_probs = probs.T[best_indices, :]
    for x_val, idx, prob, i in zip(x_values, best_indices, trans_probs, range(max_display)):
        scatter = plt.scatter(
            x_val, prob, color=colors[i], s=size, marker="_", alpha=0.1, label=labels[idx]
        )
        legend_handles.append(scatter)

    plt.ylim(0, 1)
    if separate_columns:
        plt.xticks(list(range(max_display)), best_indices)
    else:
        plt.xticks([1], "")

    if img_path is not None:
        img = plt.imread(img_path)
        ax = plt.gca()
        tick_labels = ax.xaxis.get_ticklabels()

        for tick in tick_labels:
            x_pos = tick.get_position() if separate_columns else tick_labels[0].get_position()
            ib = OffsetImage(img, zoom=img_size / img.shape[0])
            ib.image.axes = ax
            ab = AnnotationBbox(ib, x_pos, frameon=False, box_alignment=(0.5, 1.2))
            ax.add_artist(ab)

    plt.ylabel("Probability")
    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
        handle.set_alpha(1)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_morph_uncertainty(
    probs: list[list],
    probs_count: int,
    img_count: int,
    labels: list,
    img_dir: str,
    max_display: Optional[int] = None,
    img_size: int = 32,
    save_path: Optional[str] = None,
    plot_title: Optional[str] = None,
):
    """
    Plot uncertainty based on morphological created images.

    :param probs (list): List of probabilities.
    :param probs_count (int): Number of probabilities.
    :param img_count (int): Number of images.
    :param labels (list): List of labels.
    :param img_dir (str): Directory containing the images.
    :param max_display (int, optional): Maximum number of images to plot. Defaults to None.
    :param img_size (int, optional): Size of the images. Defaults to 32.
    :param save_path (str, optional): Filepath to save the plot. Defaults to None.
    :param plot_title (str, optional): Title of the plot. Defaults to None.
    :return: None
    """
    num_labels = len(labels)
    max_display = max_display if max_display else num_labels
    best_indices = list(range(num_labels))
    image_categories = img_dir.split("-morph-")

    if max_display is not None:
        sums = np.argsort(np.sum(probs.T, axis=(1, 2)))[::-1]
        best_indices = sums[:max_display]

    legend_handles = []
    scatter_size = 5000
    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_display))

    plt.figure(figsize=(20, 10))
    if plot_title is None:
        plt.title(f"{image_categories[0]} into {image_categories[1]}")
    else:
        plt.title(plot_title)
    
    plt.ylim(0, 1.1)
    plt.xticks(list(range(img_count)), [""] * img_count)
    plt.ylabel("Probability")
    ax = plt.gca()
    tick_labels = ax.xaxis.get_ticklabels()

    for step in range(img_count):
        trans_probs = probs[step].T[best_indices, :]
        for val, prob, i in zip(best_indices, trans_probs, range(max_display)):
            scatter = plt.scatter(
                [step] * probs_count,
                prob,
                color=colors[i],
                s=scatter_size,
                marker="_",
                alpha=0.1,
                label=labels[val],
            )

            if step == 0:
                legend_handles.append(scatter)

        image = plt.imread(f"{img_dir}/{step}.png")
        x_position = tick_labels[step].get_position()
        image_box = OffsetImage(image, zoom=img_size / image.shape[0])
        image_box.image.axes = ax
        annotation_box = AnnotationBbox(image_box, x_position, frameon=False, box_alignment=(0.5, 1.2))
        ax.add_artist(annotation_box)

    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
        handle.set_alpha(1)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
