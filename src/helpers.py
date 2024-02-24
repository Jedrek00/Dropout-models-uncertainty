import torchvision.transforms as transforms
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def torch_softmax(values):
    return softmax(values, dim=1)

def morph(imageA: str, imageB: str, filepath: str, steps_count: int = 10):
    """
    Morphs two images together by gradually blending them based on the given number of steps.

    :param imageA (str): Path to the first image.
    :param imageB (str): Path to the second image.
    :param filepath (str): Path to the directory where the morphed images will be saved.
    :param steps_count (int, optional): Number of steps for the morphing process. Defaults to 10.
    """
    imgA, imgB = plt.imread(imageA), plt.imread(imageB)
    imgAName = imageA.split(os.sep)[-1].split('.')[0]
    imgBName = imageB.split(os.sep)[-1].split('.')[0]

    filepath = f"{filepath}/{imgAName}-morph-{imgBName}"
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for i, step in enumerate(np.linspace(0, 1, steps_count)):
        avg_image = imgA * (1-step) + imgB * step
        avg_image = Image.fromarray((avg_image * 255).astype(np.uint8))
        avg_image.save(f"{filepath}/{i}.png")
