import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


OPENI_IMAGE_DIR  = os.path.join(config.DATA_DIR, "openi", "images")
OPENI_REPORT_DIR = os.path.join(config.DATA_DIR, "openi", "ecgen-radiology")


def parse_openi_reports(report_dir: str) -> pd.DataFrame:
    """
    Parses OpenI XML reports.
    Returns a DataFrame with columns: uid, impression, findings, image_ids.
    """
    records = []
    for xml_file in glob.glob(os.path.join(report_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        uid = os.path.splitext(os.path.basename(xml_file))[0]

        impression = ""
        findings   = ""
        for abstract_text in root.iter("AbstractText"):
            label = abstract_text.get("Label", "")
            text  = (abstract_text.text or "").strip()
            if label == "IMPRESSION":
                impression = text
            elif label == "FINDINGS":
                findings = text

        image_ids = [
            img.get("id", "") for img in root.iter("parentImage")
        ]

        records.append({
            "uid":        uid,
            "impression": impression,
            "findings":   findings,
            "image_ids":  image_ids,
            "report":     f"FINDINGS: {findings} IMPRESSION: {impression}",
        })
    return pd.DataFrame(records)


class OpenIDataset(Dataset):
    """
    Pairs a chest X-ray image with its radiology report.
    Returns (image_tensor, input_ids, attention_mask) for multimodal training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str = OPENI_IMAGE_DIR,
        tokenizer_name: str = config.TEXT_MODEL_NAME,
        max_len: int = config.MAX_TEXT_LEN,
        image_size: int = config.IMAGE_SIZE,
        split: str = "train",
    ):
        self.records   = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.max_len   = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────
        img_id  = row["image_ids"][0] if row["image_ids"] else row["uid"]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # ── Text ──────────────────────────────────────────────────────────
        enc = self.tokenizer(
            row["report"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask


def get_openi_loaders(
    batch_size: int = config.BATCH_SIZE,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = config.SEED,
) -> dict:
    df = parse_openi_reports(OPENI_REPORT_DIR)
    df = df[df["image_ids"].apply(len) > 0].reset_index(drop=True)

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(df))
    n    = len(df)
    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    loaders = {}
    workers = 2
    if os.name == "nt":
        workers = 0
    for split, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ds = OpenIDataset(df.iloc[indices], split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
