import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from densenet import DenseNet
from dataset import Dataset


DATA_PATH = "data"
# cifar or fashion
DATASET = "cifar"
IMG_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
NUM_OF_CLASSES = 10
DROPOUT_PROB = 0.2


def train_model(model: torch.nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str, test_labels: np.ndarray):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for epoch in range(EPOCHS):
        with tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
            tepoch.set_description(f"[Epoch {epoch+1}] Training:")
            for i, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                preds = outputs.cpu().max(1).indices.numpy()
                if i % 50 == 0:
                    tepoch.set_postfix(loss=loss.item(), accuracy=np.equal(preds, labels.cpu().numpy()).mean())
        with torch.no_grad():
            all_preds = []
            model.eval()
            with tqdm(test_dataloader, unit="batch", total=len(test_dataloader)) as tepoch:
                tepoch.set_description(f"[Epoch {epoch+1}] Validation:")
                for i, (inputs, labels) in enumerate(tepoch):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    preds = outputs.cpu().max(1).indices.numpy()
                    all_preds.extend(preds)
                    if i % 50 == 0:
                        tepoch.set_postfix(loss=loss.item(), accuracy=np.equal(preds, labels.cpu().numpy()).mean())
            print(f"[Epoch {epoch+1}] Accuracy on test data: {np.equal(all_preds, test_labels).mean()}")


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
    model.to(device)
    train_model(model, trainloader, testloader, device, test_labels)

if __name__ == "__main__":
    main()