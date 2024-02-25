import os
import argparse
import numpy as np

from dataset import Dataset
from train_loop import multiple_predictions
from plots import plot_morph_uncertainty
from helpers import morph


DATA_DIR = "data/test_data"
MORPH_DIR = "data/images"


def test(model_path: str, dataset: str, image_a: str, image_b: str, morph_steps: int, repeat_count: int, device: str):
    """
    Create plot showing uncertainty in model predictions.
    
    :param model_path: path to the model which will be used to make predictions.
    :param dataset: Name of the dataset from which images are taken, valid options: "cifar10" and "fashion_mnist.
    :param image_a: Name of the first file with image from "data/test_data/{dataset} location.
    :param image_b: Name of the second file with image from "data/test_data/{dataset} location.
    :param morph_steps: Number of steps during morph operation.
    :param repeat_count: How many times prediction for one morphed image should be done.
    :return: None
    """
    dataset_dir = "cifar10" if dataset == "cifar10" else "fashion_mnist"
    morph_dir_name = image_a.split(".")[0] + "-morph-" + image_b.split(".")[0]
    morph(
        os.path.join(DATA_DIR, dataset_dir, image_a),
        os.path.join(DATA_DIR, dataset_dir, image_b),
        os.path.join(MORPH_DIR, dataset_dir),
        steps_count=morph_steps,
    )

    dataset = Dataset(type=dataset)
    labels_names = dataset.train_dataset.classes

    predictions = []
    for i in range(morph_steps):
        file_path = os.path.join(MORPH_DIR, dataset_dir, morph_dir_name, f"{i}.png")
        # predictions.append([predict(model_path, file_path) for _ in range(repeat_count)])
        predictions.append(multiple_predictions(model_path, file_path, device, repeat_count))

    plot_morph_uncertainty(
        np.array(predictions),
        probs_count=repeat_count,
        img_count=morph_steps,
        labels=labels_names,
        img_dir=os.path.join(MORPH_DIR, dataset_dir, morph_dir_name),
        max_n=3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model which will be used to make predictions.')
    parser.add_argument('--dataset', type=str, choices=["cifar10", "fashion_mnist"], help='Name of the dataset from which images are taken, valid options: "cifar10" and "fashion_mnist"')
    parser.add_argument('--image_a', type=str, help='Name of the first file with image from "data/test_data/{dataset} location."')
    parser.add_argument('--image_b', type=str, help='Name of the second file with image from "data/test_data/{dataset} location."')
    parser.add_argument('--morph_steps', type=int, help='Number of steps during morph operation.')
    parser.add_argument('--repeat_count', type=int, help='How many times prediction for one morphed image should be done.')
    parser.add_argument('--device', type=str, help='Pass "cpu" to use CPU, or GPU name to train on GPU.')
    args = parser.parse_args()
    args = vars(args)
    test(**args)
    