import torchvision.transforms as transforms
from torch.nn.functional import softmax

transform = transforms.Compose(
    [
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ]
)

def torch_softmax(values):
    return softmax(values, dim=1)
