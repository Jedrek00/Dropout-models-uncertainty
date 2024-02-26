import os
import argparse
import json
import mlflow
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from convnet import ConvNet
from densenet import DenseNet
from train_loop import train_model
from visualizations import plot_history, plot_confusion_matrix

PLOTS_PATH = "plots"


def track_model_training(params: dict):
    """
    Run experiment with training model on given dataset. At the end of the training model will be saved.
    
    :param params: dictionary with the parameters. All requested parameters are listed in params/params_template.json file.
    :return: None
    """
    torch.manual_seed(params["random_seed"])
    
    device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
    print(f"{device} will be used for training")

    dataset = Dataset(type=params["dataset_name"])
    num_channels = dataset.num_channels
    img_size = dataset.img_size
    trainloader = DataLoader(
        dataset.train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    testloader = DataLoader(
        dataset.test_dataset, batch_size=params["batch_size"], shuffle=False
    )
    test_labels = np.array(dataset.test_dataset.targets)
    labels_names = dataset.train_dataset.classes

    print(f"Number of batches in train set: {len(trainloader)}")
    print(f"Number of batches in test set: {len(testloader)}")

    if params["model_architecture"] == "densenet":
        model = DenseNet(
            input_dim=img_size * img_size * num_channels,
            output_dim=params["num_of_classes"],
            hidden_dims=params["densenet"]["layers"],
            dropout_type=params["dropout_type"],
            dropout_rate=params["dropout_rate"],
        )
    elif params["model_architecture"] == "convnet":
        model = ConvNet(
            image_channels=num_channels,
            image_size=img_size,
            filters=params["convnet"]["filters"],
            kernel_sizes=params["convnet"]["kerenel_sizes"],
            dropout_type=params["dropout_type"],
            dropout_rate=params["dropout_rate"]
        )
    else:
        print(f'No such model architecture as "{params["model_architecture"]}"!')
        return

    model.to(device)

    if params["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)

    mlflow.set_experiment("Dropout for uncertainty estimation")

    with mlflow.start_run() as run:
        # saving tags
        mlflow.set_tag("dataset", params["dataset_name"])
        mlflow.set_tag("trainset_len", len(dataset.train_dataset.data))
        mlflow.set_tag("testset_len", len(dataset.test_dataset.data))
        if isinstance(model, DenseNet):
            mlflow.set_tag("model_type", "Dense")
        else:
            mlflow.set_tag("model_type", "CNN")
        mlflow.set_tag("optimizer", optimizer.__class__.__name__)
        mlflow.set_tag("dropout", params["dropout_type"])
        mlflow.set_tag("random_seed", params["random_seed"])

        # saving parameters
        mlflow.log_param("lr", params["lr"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("epochs", params["epochs"])
        mlflow.log_param("dropout_rate", params["dropout_rate"])

        history, model = train_model(
            model, optimizer, trainloader, testloader, params["epochs"], device, test_labels
        )
        dir_path = os.path.join(PLOTS_PATH, run.info.run_name)
        os.makedirs(dir_path)
        plot_history(history, os.path.join(dir_path, "acc_loss.png"))
        plot_confusion_matrix(
            history["val_preds"],
            test_labels,
            labels_names,
            os.path.join(dir_path, "confusion_matrix.png"),
        )

        # saving metrics
        mlflow.log_metric("train_loss", history["train_loss"][-1])
        mlflow.log_metric("val_loss", history["val_loss"][-1])
        mlflow.log_metric("train_acc", history["train_acc"][-1])
        mlflow.log_metric("val_acc", history["val_acc"][-1])

        # saving plots
        mlflow.log_artifact(os.path.join(dir_path, "acc_loss.png"))
        mlflow.log_artifact(os.path.join(dir_path, "confusion_matrix.png"))
        
        # saving model
        model_path = os.path.join(
            "models",
            params["dataset_name"],
            f'{params["model_architecture"]}-{params["dropout_type"]}-{run._info.run_name}.pt',
        )
        torch.save(model, model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, help='path to the JSON file with (hiper)parameters used for model training.')
    args = parser.parse_args()
    
    with open(args.param_file, "r") as f:
        parameters = json.load(f)
    
    track_model_training(parameters)
