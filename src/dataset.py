import os
from typing import Union
from torchvision import datasets
import torchvision.transforms as transforms


class Dataset:
    types = {
        "cifar": {"directory": "cifar-10-batches-py", "instance": datasets.CIFAR10},
        "fashion": {"directory": "FashionMNIST", "instance": datasets.FashionMNIST},
    }

    transform = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __init__(self, type="cifar"):
        self.path = "../data"
        self.dataset_info = Dataset.types[type]
        self.dataset_path = os.path.join(self.path, self.dataset_info['directory'])

        self.train_dataset = self.create_dataset(train=True)
        self.test_dataset = self.create_dataset(train=False)

    def create_dataset(self, train: bool) -> Union[datasets.CIFAR10, datasets.FashionMNIST]:
        return self.dataset_info["instance"](
            root=self.path, train=train, download=True, transform=Dataset.transform
        )
