import numpy as np
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ]
)

def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(values) / np.exp(values).sum()

def stable_softmax(values: np.ndarray) -> np.ndarray:
    e_x = np.exp(values - np.max(values))
    return e_x / e_x.sum()