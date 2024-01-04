import os
import mlflow
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import hiperparameters
from dataset import Dataset
from convnet import ConvNet
from densenet import DenseNet
from plots import (
    plot_history,
    plot_confusion_matrix,
    plot_uncertainty,
)
from helpers import transform, torch_softmax


DATA_PATH = "data"
PLOTS_PATH = "plots"
TEST_CIFAR_PATH = "data/test_data/cifar"
TEST_FASHION_PATH = "data/test_data/fashion"





def train_model(
    model: torch.nn.Module,
    optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: str,
    test_labels: np.ndarray,
):
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
    for epoch in range(hiperparameters.EPOCHS):
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
                # torch.save(model, os.path.join(MODELS_PATH, "model3.pt"))
    return history, best_model


def predict(model_path: str, image_path: str) -> np.ndarray:
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.to("cpu")
    model.train()
    img = Image.open(image_path)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        logits = model(img)
        probs = torch_softmax(logits)
    return probs.numpy()[0]


def main(model_architecture, dataset_name, dropout_type, dropout_rate, random_seed):
    torch.manual_seed(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} will be used for training")

    dataset = Dataset(type=dataset_name)
    NUM_CHANNELS = dataset.num_channels
    IMG_SIZE = dataset.img_size
    trainloader = DataLoader(dataset.train_dataset, batch_size=hiperparameters.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset.test_dataset, batch_size=hiperparameters.BATCH_SIZE, shuffle=False)
    test_labels = np.array(dataset.test_dataset.targets)
    labels_names = dataset.train_dataset.classes

    print(f"Number of batches in train set: {len(trainloader)}")
    print(f"Number of batches in test set: {len(testloader)}")

    # model = torch.load(os.path.join(MODELS_PATH, "model.pt"))
    if model_architecture == 'densenet':
        model = DenseNet(
            IMG_SIZE * IMG_SIZE * NUM_CHANNELS,
            hiperparameters.NUM_OF_CLASSES,
            [512, 256, 128],
            dropout_rate,
            dropout_type,
        )
    elif model_architecture == 'convnet':
        model = ConvNet(image_channels=NUM_CHANNELS, image_size=IMG_SIZE, filters=[32, 64, 128], kernel_sizes=[(3, 3), (3, 3), (3, 3)], dropout_type=dropout_type, dropout_rate=dropout_rate)
    else:
        print(f'No such model architecture as "{model_architecture}"!')
        return

    if not model.valid:
        return

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hiperparameters.LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=hiperparameters.LR, momentum=0.9)

    mlflow.set_experiment("Dropout for uncertainty estimation")

    with mlflow.start_run() as run:
        # saving tags
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag("trainset_len", len(dataset.train_dataset.data))
        mlflow.set_tag("testset_len", len(dataset.test_dataset.data))
        if isinstance(model, DenseNet):
            mlflow.set_tag("model_type", "Dense")
        else:
            mlflow.set_tag("model_type", "CNN")
        mlflow.set_tag("optimizer", optimizer.__class__.__name__)
        mlflow.set_tag("dropout", dropout_type)
        mlflow.set_tag("random_seed", random_seed)

        # saving parameters
        mlflow.log_param("lr", hiperparameters.LR)
        mlflow.log_param("batch_size", hiperparameters.BATCH_SIZE)
        mlflow.log_param("epochs", hiperparameters.EPOCHS)
        mlflow.log_param("dropout_rate", dropout_rate)
        # TODO save number and sizes of layers for desne and filters for CNN

        history, model = train_model(
            model, optimizer, trainloader, testloader, device, test_labels
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

        # TODO save models
        model_path = os.path.join('models', dataset_name, f'{model_architecture}-{dropout_type}-{dropout_rate}', f'random-seed-{random_seed}.pt')
        torch.save(model, model_path)

if __name__ == "__main__":
    for random_seed in hiperparameters.RANDOM_SEEDS:
        for dataset_name in hiperparameters.DATASETS:
            for model_architecture in hiperparameters.MODEL_ARCHITECTURES:
                for dropout_type in hiperparameters.DROPOUT_TYPES:
                    for dropout_rate in hiperparameters.DROPOUTS_RATES:
                        print(f'\nTraining:\nRandom seed: {random_seed}\nDataset: {dataset_name}\nModel architecture: {model_architecture}\nDropout type: {dropout_type}\nDropout rate: {dropout_rate}')
                        main(model_architecture=model_architecture, dataset_name=dataset_name, dropout_type=dropout_type, dropout_rate=dropout_rate, random_seed=random_seed)

    # RANDOM_SEED = '100'
    # DATASET = 'fashion_mnist'
    # MODEL = 'convnet'
    # DROPOUT_TYPE = 'standard'
    # DROPOUT_RATE = '0.5'
    # MODELS_PATH = os.path.join('models', DATASET, f'{MODEL}-{DROPOUT_TYPE}-{DROPOUT_RATE}',
    #                           f'random-seed-{RANDOM_SEED}.pt')
    # dataset = Dataset(type=DATASET)
    # labels_names = dataset.train_dataset.classes
    # p = []
    # for i in range(100):
    #     p.append(
    #         predict(
    #             MODELS_PATH,
    #             os.path.join(TEST_FASHION_PATH, "0.png"),
    #         )
    #     )
    # plot_uncertainty(
    #     np.array(p),
    #     labels=labels_names,
    #     max_n=4,
    #     separate=True,
    #     img_path=os.path.join(TEST_FASHION_PATH, "0.png"),
    # )
