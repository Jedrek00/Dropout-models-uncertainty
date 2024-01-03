import os
from typing import Union
from torchvision import datasets

from helpers import transform


class Dataset:
    types = {
        "cifar10": {"directory": "cifar-10-batches-py", "instance": datasets.CIFAR10},
        "fashion_mnist": {"directory": "FashionMNIST", "instance": datasets.FashionMNIST},
    }


    def __init__(self, type="cifar"):
        self.path = "../data"
        self.dataset_info = Dataset.types[type]
        self.dataset_path = os.path.join(self.path, self.dataset_info['directory'])

        self.train_dataset = self.create_dataset(train=True)
        self.test_dataset = self.create_dataset(train=False)

        self.num_channels = len(self.train_dataset[0][0])
        self.img_size = len(self.train_dataset[0][0][0])

    def create_dataset(self, train: bool) -> Union[datasets.CIFAR10, datasets.FashionMNIST]:
        return self.dataset_info["instance"](
            root=self.path, train=train, download=True, transform=transform
        )
