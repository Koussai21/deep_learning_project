import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import ChestMNIST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(split: str, image_size: int = config.IMAGE_SIZE):
    """
    Augmentation during training, deterministic at eval time.
    All values normalised with ImageNet statistics (standard for transfer learning).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        normalize,
    ])


def get_chest_mnist_loaders(
    image_size: int = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 2,
    data_dir: str = config.DATA_DIR,
):
    """
    Returns train / val / test DataLoaders for ChestMNIST.
    Labels are multi-hot vectors of shape (14,) cast to float for BCE loss.
    """
    os.makedirs(data_dir, exist_ok=True)

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = ChestMNIST(
            split=split,
            transform=get_transforms(split, image_size),
            download=True,
            root=data_dir,
            size=image_size,
        )
        workers = num_workers
        if os.name == "nt":
            # Windows can fail when pickling worker state during DataLoader startup.
            workers = 0
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


class MultiLabelWrapper(torch.utils.data.Dataset):
    """Wraps a MedMNIST dataset, converting int labels to float multi-hot vectors."""

    def __init__(self, base_dataset, num_classes: int = config.NUM_CLASSES):
        self.dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # label shape: (num_classes,) with 0/1 integers
        label = torch.tensor(label, dtype=torch.float32).squeeze()
        return image, label


def get_class_weights(loader: DataLoader, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """
    Computes positive class frequency to build a pos_weight tensor for BCE.
    Higher weight for rare classes counters label imbalance.
    """
    pos_count = torch.zeros(num_classes)
    total = 0
    for _, labels in loader:
        pos_count += labels.sum(dim=0)
        total += labels.shape[0]
    neg_count = total - pos_count
    # avoid division by zero
    pos_weight = neg_count / (pos_count + 1e-6)
    return pos_weight
