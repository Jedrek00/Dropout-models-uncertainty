from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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