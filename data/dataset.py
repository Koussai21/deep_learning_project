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
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        pad = int(image_size * 0.1)
        return transforms.Compose([
            transforms.Resize((image_size + pad, image_size + pad)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
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
    num_workers: int = 4,
    data_dir: str = config.DATA_DIR,
    augment_train: bool = True,
):
    os.makedirs(data_dir, exist_ok=True)

    loaders = {}
    for split in ("train", "val", "test"):
        tfm_split = split if augment_train else "val"
        dataset = ChestMNIST(
            split=split,
            transform=get_transforms(tfm_split, image_size),
            download=True,
            root=data_dir,
            size=image_size,
        )
        workers = num_workers
        if os.name == "nt":
            workers = 0
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(workers > 0),
        )
    return loaders


class MultiLabelWrapper(torch.utils.data.Dataset):

    def __init__(self, base_dataset, num_classes: int = config.NUM_CLASSES):
        self.dataset = base_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = torch.tensor(label, dtype=torch.float32).squeeze()
        return image, label


def get_class_weights(
    loader: DataLoader,
    num_classes: int = config.NUM_CLASSES,
    max_weight: float = 10.0,
) -> torch.Tensor:
    pos_count = torch.zeros(num_classes)
    total = 0
    for _, labels in loader:
        pos_count += labels.sum(dim=0)
        total += labels.shape[0]
    neg_count = total - pos_count
    pos_weight = neg_count / (pos_count + 1e-6)
    return torch.clamp(pos_weight, max=max_weight)
