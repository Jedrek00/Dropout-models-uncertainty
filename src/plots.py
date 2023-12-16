from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.image as mpimg


def plot_history(history: dict, filepath: Optional[str] = None):
    x = range(len(history['train_loss']))
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.plot(x, history['train_loss'], label='train_loss')
    plt.plot(x, history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(212)
    plt.plot(x, history['train_acc'], label='train_acc')
    plt.plot(x, history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    if filepath is not None:
        plt.savefig(filepath)


def plot_confusion_matrix(preds: list, labels: list, class_names: list, filepath: Optional[str] = None):
    plt.figure(figsize=(12, 12))
    matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    disp.plot(xticks_rotation="vertical")
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath)

# input: (100, 10)
def plot_uncertainty(probs: list, max_n: Optional[int] = None, separate: bool = True, img_path: Optional[str] = None, img_size: int = 32, filepath: Optional[str] = None):
    CONST_I = 1
    max_n = max_n if max_n else 10
    best_n = list(range(10))

    if max_n != None:
        sums = np.argsort(np.sum(probs.T, axis=1))[::-1]
        best_n = sums[:max_n]

    x = [CONST_I] * probs.shape[0]
    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_n))
    legend_handles = []


    new_probs = probs.T[best_n, :]
    for val, p, i in zip(best_n, new_probs, list(range(max_n))):
        scatter = plt.scatter(x, p, color=colors[i], s=50000, marker='_', alpha=0.3, label=f"value: {val + 1}")
        legend_handles.append(scatter)

    plt.ylim(0, 1)
    plt.xlim(0,2)
    # plt.xticks([1], "")
    if img_path != None:
        img = plt.imread(img_path)
        ax = plt.gca()
        tick_labels = ax.xaxis.get_ticklabels()
        ib = OffsetImage(img, zoom=img_size / img.shape[0])
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[0].get_position(),
                        frameon=False,
                        box_alignment=(0.5, 1.2)
                        )
        ax.add_artist(ab)
    plt.ylabel("probability")
    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
    
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


def plot_uncertainty_separate(probs: list, labels: list, max_n: Optional[int] = None, img_path: Optional[str] = None, img_size: int = 32, filepath: Optional[str] = None):
    n = len(labels)
    max_n = max_n if max_n else n
    best_n = list(range(n))

    if max_n != None:
        sums = np.argsort(np.sum(probs.T, axis=1))[::-1]
        best_n = sums[:max_n]

    x = [[i] * probs.shape[0] for i in range(len(best_n))]
    colors = plt.cm.get_cmap("jet")(np.linspace(0, 1, max_n))
    legend_handles = []


    new_probs = probs.T[best_n, :]
    for x_p, val, p, i in zip(x, best_n, new_probs, list(range(max_n))):
        scatter = plt.scatter(x_p, p, color=colors[i], s=25000 / max_n, marker='_', linewidths=3, alpha=0.1, label=labels[val])
        legend_handles.append(scatter)

    plt.ylim(0, 1)
    plt.xticks(list(range(max_n)), best_n)
    if img_path != None:
        img = plt.imread(img_path)
        ax = plt.gca()
        tick_labels = ax.xaxis.get_ticklabels()

        for tick in tick_labels:
            ib = OffsetImage(img, zoom=img_size / img.shape[0])
            ib.image.axes = ax
            ab = AnnotationBbox(ib,
                            tick.get_position(),
                            frameon=False,
                            box_alignment=(0.5, 1.2)
                            )
            ax.add_artist(ab)
    plt.ylabel("probability")
    legend = plt.legend(handles=legend_handles)
    for handle in legend.legend_handles:
        handle.set_sizes([100])
    
    if filepath != None:
        plt.savefig(filepath)
    plt.show()