import os
from typing import Optional
import numpy as np
from tqdm import tqdm
import mlflow
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from convnet import ConvNet
from densenet import DenseNet


DATA_PATH = "data"
PLOTS_PATH = "plots"
MODELS_PATH = "models"
# cifar or fashion
DATASET = "cifar"
# DATASET = "fashion"
IMG_SIZE = 32
# IMG_SIZE = 28
NUM_CHANNELS = 3
# NUM_CHANNELS = 1
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.001
NUM_OF_CLASSES = 10
DROPOUT_PROB = 0.2
DROPOUT_TYPE = "standard"


def plot_history(history: dict, filename: Optional[str] = None):
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
    if filename is not None:
        plt.savefig(os.path.join(PLOTS_PATH, filename))
    plt.show()


def train_model(model: torch.nn.Module, optimizer, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str, test_labels: np.ndarray):
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_acc = 0
    for epoch in range(EPOCHS):
        with tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
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
                    tepoch.set_postfix(loss=loss.item(), accuracy=np.equal(preds, labels.cpu().numpy()).mean())
            history['train_loss'].append(np.mean(train_loss_epoch))
            history['train_acc'].append(np.mean(train_acc_epoch))
        with torch.no_grad():
            all_preds = []
            val_loss_epoch = []
            model.eval()
            with tqdm(test_dataloader, unit="batch", total=len(test_dataloader)) as tepoch:
                tepoch.set_description(f"[Epoch {epoch+1}] Validation:")
                for i, (inputs, labels) in enumerate(tepoch):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    preds = outputs.cpu().max(1).indices.numpy()
                    all_preds.extend(preds)
                    val_loss_epoch.append(loss.item())
                    if i % 50 == 0:
                        tepoch.set_postfix(loss=loss.item(), accuracy=np.equal(preds, labels.cpu().numpy()).mean())
            acc = np.equal(all_preds, test_labels).mean()
            history['val_loss'].append(np.mean(val_loss_epoch))
            history['val_acc'].append(acc)
            print(f"[Epoch {epoch+1}] Accuracy on test data: {acc}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model, os.path.join(MODELS_PATH, "model2.pt"))
    return history

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{device} will be used for training")

    dataset = Dataset(type=DATASET)
    trainloader = DataLoader(dataset.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_labels = np.array(dataset.test_dataset.targets)

    print(f"Number of batches in train set: {len(trainloader)}")
    print(f"Number of batches in test set: {len(testloader)}")

    # model = torch.load(os.path.join(MODELS_PATH, "model.pt"))
    # model = DenseNet(IMG_SIZE*IMG_SIZE*NUM_CHANNELS, NUM_OF_CLASSES, [512, 256, 128], DROPOUT_PROB, DROPOUT_TYPE)
    model = ConvNet(image_channels=NUM_CHANNELS, image_size=IMG_SIZE, filters=[32, 64, 128], kernel_sizes=[(3, 3), (3, 3), (3, 3)], dropout_type=DROPOUT_TYPE)
    # model = ConvNet(image_channels=NUM_CHANNELS, image_size=IMG_SIZE, filters=[32, 64, 128], kernel_sizes=[(3, 3), (3, 3), (3, 3)], dropout_type=DROPOUT_TYPE, dropout_rate=0.5)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    mlflow.set_experiment("Dropout for uncertainty estimation")

    with mlflow.start_run() as run:
        # saving tags
        mlflow.set_tag("dataset", DATASET)
        mlflow.set_tag("trainset_len", len(dataset.train_dataset.data))
        mlflow.set_tag("testset_len", len(dataset.test_dataset.data))
        if isinstance(model, DenseNet):
            mlflow.set_tag("model_type", "Dense")
        else:
            mlflow.set_tag("model_type", "CNN")
        mlflow.set_tag("optimizer", optimizer.__class__.__name__)
        mlflow.set_tag("dropout", DROPOUT_TYPE)

        # saving parameters
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("dropout_rate", DROPOUT_PROB)
        # TODO save number and sizes of layers for desne and filters for CNN
        
        history = train_model(model, optimizer, trainloader, testloader, device, test_labels)
        plot_name = run.info.run_name + ".png"
        plot_history(history, filename=plot_name)

        # saving metrics
        mlflow.log_metric("train_loss", history["train_loss"][-1])
        mlflow.log_metric("val_loss", history["val_loss"][-1])
        mlflow.log_metric("train_acc", history["train_acc"][-1])
        mlflow.log_metric("val_acc", history["val_acc"][-1])

        # saving plot
        mlflow.log_artifact(os.path.join(PLOTS_PATH, plot_name))

        # TODO save models


if __name__ == "__main__":
    main()
