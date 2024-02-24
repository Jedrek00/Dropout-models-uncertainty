import os
from typing import Union
from torchvision import datasets

from helpers import transform

PATH = "../data"

class DatasetTypeException(Exception):
    pass

class Dataset:
    TYPES = {
        "cifar10": {"directory": "cifar-10-batches-py", "instance": datasets.CIFAR10},
        "fashion_mnist": {"directory": "FashionMNIST", "instance": datasets.FashionMNIST},
    }

    def __init__(self, type="cifar10"):
        if type not in self.TYPES:
                raise DatasetTypeException(f"There are two valid datasets: 'cifar10' and 'fashion_mnist', you passed {type}!")
        self.dataset_info = Dataset.TYPES[type]
        self.dataset_path = os.path.join(PATH, self.dataset_info['directory'])

        self.train_dataset = self.create_dataset(train=True)
        self.test_dataset = self.create_dataset(train=False)

        self.num_channels = len(self.train_dataset[0][0])
        self.img_size = len(self.train_dataset[0][0][0])

    def create_dataset(self, train: bool) -> Union[datasets.CIFAR10, datasets.FashionMNIST]:
        return self.dataset_info["instance"](
            root=PATH, train=train, download=True, transform=transform
        )
