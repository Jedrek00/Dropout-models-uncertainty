import os
import mlflow
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from convnet import ConvNet
from densenet import DenseNet
from plots import plot_history, plot_confusion_matrix
from helpers import transform, softmax


DATA_PATH = "data"
PLOTS_PATH = "plots"
MODELS_PATH = "models"

RANDOM_SEED = 69

# cifar or fashion
DATASET = "cifar"
# DATASET = "fashion"
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
NUM_OF_CLASSES = 10
DROPOUT_PROB = 0.2
DROPOUT_TYPE = "standard"


def train_model(model: torch.nn.Module, optimizer, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str, test_labels: np.ndarray):
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_preds': []
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
            history['val_preds'] = all_preds
            history['val_loss'].append(np.mean(val_loss_epoch))
            history['val_acc'].append(acc)
            print(f"[Epoch {epoch+1}] Accuracy on test data: {acc}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model, os.path.join(MODELS_PATH, "model2.pt"))
    return history


def predict(model_path: str, image_path: str, drop_rate: float):    
    model = torch.load(model_path)
    model.to("cpu")
    img = Image.open(image_path)  
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        logits = model(img, test_dropout_rate=drop_rate)
        logits = logits.numpy()
        probs = softmax(logits)
    return logits, probs


def main():
    torch.manual_seed(RANDOM_SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{device} will be used for training")

    dataset = Dataset(type=DATASET)
    NUM_CHANNELS = dataset.num_channels
    IMG_SIZE = dataset.img_size
    trainloader = DataLoader(dataset.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_labels = np.array(dataset.test_dataset.targets)
    labels_names = dataset.train_dataset.classes


    print(f"Number of batches in train set: {len(trainloader)}")
    print(f"Number of batches in test set: {len(testloader)}")

    # model = torch.load(os.path.join(MODELS_PATH, "model.pt"))
    model = DenseNet(IMG_SIZE*IMG_SIZE*NUM_CHANNELS, NUM_OF_CLASSES, [512, 256, 128], DROPOUT_PROB, DROPOUT_TYPE)
    # model = ConvNet(image_channels=NUM_CHANNELS, image_size=IMG_SIZE, filters=[32, 64, 128], kernel_sizes=[(3, 3), (3, 3), (3, 3)], dropout_type=DROPOUT_TYPE)
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
        mlflow.set_tag('random_seed', RANDOM_SEED)

        # saving parameters
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("dropout_rate", DROPOUT_PROB)
        # TODO save number and sizes of layers for desne and filters for CNN
        
        history = train_model(model, optimizer, trainloader, testloader, device, test_labels)
        dir_path = os.path.join(PLOTS_PATH, run.info.run_name)
        os.makedirs(dir_path)
        plot_history(history, os.path.join(dir_path, "acc_loss.png"))
        plot_confusion_matrix(history["val_preds"], test_labels, labels_names, os.path.join(dir_path, "confusion_matrix.png"))

        # saving metrics
        mlflow.log_metric("train_loss", history["train_loss"][-1])
        mlflow.log_metric("val_loss", history["val_loss"][-1])
        mlflow.log_metric("train_acc", history["train_acc"][-1])
        mlflow.log_metric("val_acc", history["val_acc"][-1])

        # saving plots
        mlflow.log_artifact(os.path.join(dir_path, "acc_loss.png"))
        mlflow.log_artifact(os.path.join(dir_path, "confusion_matrix.png"))

        # TODO save models

if __name__ == "__main__":
    main()
    l, p = predict(os.path.join(MODELS_PATH, "model2.pt"), os.path.join(DATA_PATH, "0002.png"), 0.2)
    print(l)
    print(p)
