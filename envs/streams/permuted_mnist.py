import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Iterator, Tuple


def make_permuted_mnist_stream(
    batch_size: int = 30, data_dir: str = "./data"
) -> Iterator[DataLoader]:
    """
    Generates an infinite stream of Permuted MNIST tasks.
    Each yield returns a DataLoader for a new task where the 784 pixels
    have been randomly permuted.

    Args:
        batch_size: The batch size for the DataLoader (Paper uses 30).
        data_dir: Directory to store the downloaded MNIST data.

    Yields:
        DataLoader: A PyTorch DataLoader for the current permutation task.
    """
    # Base transform: ToTensor flattens to [1, 28, 28] and scales to [0, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
        ]
    )

    # Download/Load the standard MNIST training set
    mnist_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    while True:
        # Generate a new random permutation for the 784 pixels
        permutation = torch.randperm(784)

        # Create a custom dataset applying the permutation on the fly
        class PermutedDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(mnist_train)

            def __getitem__(self, idx):
                img, target = mnist_train[idx]
                return img[permutation], target

        permuted_dataset = PermutedDataset()

        # Paper uses batch size 30, 1 pass through the data (2000 updates)
        yield DataLoader(permuted_dataset, batch_size=batch_size, shuffle=True)
