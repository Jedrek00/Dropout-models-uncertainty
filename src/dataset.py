import os

from torchvision import datasets


class Dataset:
    types = {
        "cifar": {"directory": "cifar-10-batches-py", "instance": datasets.CIFAR10},
        "fashion": {"directory": "FashionMNIST", "instance": datasets.FashionMNIST},
    }

    def __init__(self, type="cifar"):
        self.path = "../data"
        self.dataset_info = Dataset.types[type]
        self.dataset_path = f"{self.path}/{self.dataset_info['directory']}"

        self.train_dataset = self.create_dataset(True)
        self.test_dataset = self.create_dataset(False)

    def create_dataset(self, train):
        return self.dataset_info["instance"](
            root=self.path, train=train, download=True, transform=None
        )
    