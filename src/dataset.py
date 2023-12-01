import os
import torchvision.datasets as datasets


class Dataset:
    def __init__(self, replace=False):
        self.path = "../data"
        self.replace = replace
        self.dataset = self.create_dataset()
        print(self.dataset)

    def create_dataset(self):
        if (
            self.replace
            or not os.path.exists(self.path)
            or not len(os.listdir(self.path)) > 0
        ):
            return datasets.CIFAR10(
                root=self.path, train=False, download=True, transform=None
            )
        else:
            if os.path.exists("../data/cifar-10-batches-py"):
                return datasets.CIFAR10(
                    root=self.path, train=False, download=False, transform=None
                )
            else:
                raise "Corrupted folder"
