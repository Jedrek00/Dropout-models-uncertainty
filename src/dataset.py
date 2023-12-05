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
        self.dataset_path = os.path.join(self.path, self.dataset_info['directory'])

        self.train_dataset = self.create_dataset(train=True)
        self.test_dataset = self.create_dataset(train=False)

    def create_dataset(self, train: bool):
        return self.dataset_info["instance"](
            root=self.path, train=train, download=True, transform=None
        )
