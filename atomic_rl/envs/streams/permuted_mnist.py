import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Iterator

from ._data_dir import get_default_data_dir


class PermutedDataset(Dataset):
    """Dataset wrapper that applies a specific permutation to the data."""

    def __init__(self, dataset: Dataset, permutation: torch.Tensor):
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img[self.permutation], target


def make_permuted_mnist_stream(
    batch_size: int = 30, data_dir: str | None = None
) -> Iterator[DataLoader]:
    if data_dir is None:
        data_dir = get_default_data_dir("mnist")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    mnist_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    while True:
        # 1. Generate a new random permutation
        permutation = torch.randperm(784)

        # 2. Wrap the base dataset with the new permutation
        permuted_dataset = PermutedDataset(mnist_train, permutation)

        # 3. Yield a dataloader that will iterate exactly once over the 60k images randomly
        yield DataLoader(permuted_dataset, batch_size=batch_size, shuffle=True)
