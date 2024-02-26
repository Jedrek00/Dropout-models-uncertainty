import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from helpers import transform, torch_softmax


def train_model(
    model: torch.nn.Module,
    optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    device: str,
    test_labels: np.ndarray,
):
    """
    Torch train loop.

    :param model: Torch model to train.
    :param optimizer: Torch optimizer.
    :param train_dataloader: Dataloader with train data.
    :param test_dataloader: Dataloader with test data.
    :param epochs: Number of epochs.
    :param device: Pass 'cpu' to use CPU, or GPU name to train on GPU.
    :param test_labels: array with correct labels for test data. 
    :return: dictionary with stats from training and best model.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_preds": [],
    }
    best_acc = 0
    best_model = deepcopy(model)
    for epoch in range(epochs):
        with tqdm(
            train_dataloader, unit="batch", total=len(train_dataloader)
        ) as tepoch:
            tepoch.set_description(f"[Epoch {epoch+1}] Training:")
            train_loss_epoch = []
            train_acc_epoch = []
            for i, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                preds = outputs.cpu().max(1).indices.numpy()
                train_loss_epoch.append(loss.item())
                train_acc_epoch.append(np.equal(preds, labels.cpu().numpy()).mean())
                if i % 50 == 0:
                    tepoch.set_postfix(
                        loss=loss.item(),
                        accuracy=np.equal(preds, labels.cpu().numpy()).mean(),
                    )
            history["train_loss"].append(np.mean(train_loss_epoch))
            history["train_acc"].append(np.mean(train_acc_epoch))
        with torch.no_grad():
            all_preds = []
            val_loss_epoch = []
            model.eval()
            with tqdm(
                test_dataloader, unit="batch", total=len(test_dataloader)
            ) as tepoch:
                tepoch.set_description(f"[Epoch {epoch+1}] Validation:")
                for i, (inputs, labels) in enumerate(tepoch):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    preds = outputs.cpu().max(1).indices.numpy()
                    all_preds.extend(preds)
                    val_loss_epoch.append(loss.item())
                    if i % 50 == 0:
                        tepoch.set_postfix(
                            loss=loss.item(),
                            accuracy=np.equal(preds, labels.cpu().numpy()).mean(),
                        )
            acc = np.equal(all_preds, test_labels).mean()
            history["val_preds"] = all_preds
            history["val_loss"].append(np.mean(val_loss_epoch))
            history["val_acc"].append(acc)
            print(f"[Epoch {epoch+1}] Accuracy on test data: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_model = deepcopy(model)
    return history, best_model


def multiple_predictions(model_path: str, image_path: str, device: str, n: int):
    """
    Make multiple prediction on the single image.

    :param model_path: Path to the trained model.
    :param image_path: Path to the image.
    :param device: Pass "cpu" to use CPU, or GPU name to train on GPU.
    :param n: how many predictions should be made on given image.
    :return: Array with the prediction made by the model in form of probabilities.
    """
    model = torch.load(model_path)
    model.to(device)
    model.train() # we want different prediction each time
    img = Image.open(image_path)
    img = transform(img)
    img = torch.unsqueeze(img, 0).to(device)
    probs_list = []
    with torch.no_grad():
        for _ in range(n):
            logits = model(img)
            probs = torch_softmax(logits)
            probs_list.append(probs.to("cpu").numpy()[0])
    return probs_list