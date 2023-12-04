import os

from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split


class Dataset:
    types = {
        "cifar": {"directory": "cifar-10-batches-py", "instance": datasets.CIFAR10},
        "fashion": {"directory": "FashionMNIST", "instance": datasets.FashionMNIST},
    }

    def __init__(self, type="cifar", test_size=0.2):
        self.path = "../data"
        self.dataset_info = Dataset.types[type]
        self.dataset_path = f"{self.path}/{self.dataset_info['directory']}"
        self.test_size = test_size

        self.dataset = self.create_dataset()
        self.train_dataset, self.test_dataset = self.dataset_split()

        print(len(self.train_dataset))
        print(len(self.test_dataset))

    def create_dataset(self):
        return self.dataset_info["instance"](
            root=self.path, train=False, download=True, transform=None
        )

    def dataset_split(self):
        train_indices, test_indices = train_test_split(
            range(len(self.dataset)), test_size=self.test_size
        )
        train = Subset(self.dataset, train_indices)
        test = Subset(self.dataset, test_indices)

        return train, test
