import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from convnet import ConvNet
from densenet import DenseNet


DATA_PATH = "data"
# cifar or fashion
DATASET = "cifar"
# DATASET = "fashion"
IMG_SIZE = 32
# IMG_SIZE = 28
NUM_CHANNELS = 3
# NUM_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
NUM_OF_CLASSES = 10
DROPOUT_PROB = 0.2


def plot_history(history: dict):
    import matplotlib.pyplot as plt
    x = range(len(history['train_loss']))
    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    plt.plot(x, history['train_loss'], label='train_loss')
    plt.plot(x, history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(122)
    plt.plot(x, history['train_acc'], label='train_acc')
    plt.plot(x, history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()


def train_model(model: torch.nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str, test_labels: np.ndarray):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
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
            history['val_loss'].append(np.mean(val_loss_epoch))
            history['val_acc'].append(np.equal(all_preds, test_labels).mean())
            print(f"[Epoch {epoch+1}] Accuracy on test data: {np.equal(all_preds, test_labels).mean()}")
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

    model = DenseNet(IMG_SIZE*IMG_SIZE*NUM_CHANNELS, NUM_OF_CLASSES, [512, 256, 128], DROPOUT_PROB)
    # model = ConvNet(image_channels=NUM_CHANNELS, use_standard_dropout=False, use_spatial_dropout=False, use_cutout_dropout=False)
    # model = ConvNet(image_channels=NUM_CHANNELS, use_standard_dropout=True, use_spatial_dropout=False, use_cutout_dropout=False, dropout_rate=0.5)
    model.to(device)
    history = train_model(model, trainloader, testloader, device, test_labels)
    plot_history(history)


if __name__ == "__main__":
    main()
