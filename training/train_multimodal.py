import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.multimodal import (
    ImageOnlyModel, TextOnlyModel,
    EarlyFusionModel, IntermediateFusionModel, LateFusionModel,
)
from evaluation.metrics import compute_metrics
from mlflow_utils.tracking import setup_mlflow, MLflowRun, log_epoch
from training.utils import set_seed, get_device, EarlyStopping, count_parameters


PATHOLOGY_KEYWORDS = {
    "Atelectasis":        ["atelecta"],
    "Cardiomegaly":       ["cardiomegaly", "enlarged heart", "heart size"],
    "Effusion":           ["effusion"],
    "Infiltration":       ["infiltrate", "infiltration"],
    "Mass":               ["mass"],
    "Nodule":             ["nodule"],
    "Pneumonia":          ["pneumonia"],
    "Pneumothorax":       ["pneumothorax"],
    "Consolidation":      ["consolidation"],
    "Edema":              ["edema"],
    "Emphysema":          ["emphysema"],
    "Fibrosis":           ["fibrosis", "fibrotic"],
    "Pleural_Thickening": ["pleural thickening"],
    "Hernia":             ["hernia"],
}


def derive_labels(report: str) -> np.ndarray:
    text = report.lower()
    vec = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    for i, name in enumerate(config.CLASS_NAMES):
        if any(kw in text for kw in PATHOLOGY_KEYWORDS[name]):
            vec[i] = 1.0
    return vec


def build_model(mode: str, fusion: str) -> nn.Module:
    if mode == "image":
        return ImageOnlyModel()
    if mode == "text":
        return TextOnlyModel()
    if mode == "fusion":
        return {
            "early":        EarlyFusionModel,
            "intermediate": IntermediateFusionModel,
            "late":         LateFusionModel,
        }[fusion]()
    raise ValueError(mode)


def forward_batch(model, mode, image, input_ids, attn, device):
    image     = image.to(device)
    input_ids = input_ids.to(device)
    attn      = attn.to(device)
    if mode == "image":
        return model(image=image)
    if mode == "text":
        return model(input_ids=input_ids, attention_mask=attn)
    return model(image=image, input_ids=input_ids, attention_mask=attn)


@torch.no_grad()
def evaluate(model, loader, mode, criterion, device):
    model.eval()
    all_logits = []
    for image, input_ids, attn in loader:
        logits = forward_batch(model, mode, image, input_ids, attn, device)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def run(args):
    set_seed()
    device = get_device()
    print(f"Device: {device} | mode: {args.mode} | fusion: {args.fusion}")

    from data.multimodal_dataset import parse_openi_reports, OpenIDataset, OPENI_REPORT_DIR
    from torch.utils.data import DataLoader

    df = parse_openi_reports(OPENI_REPORT_DIR)
    df = df[df["image_ids"].apply(len) > 0].reset_index(drop=True)
    df["labels"] = df["report"].apply(derive_labels)
    print(f"Parsed {len(df)} image-report pairs.")

    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(df))
    n_test, n_val = int(0.15 * len(df)), int(0.15 * len(df))
    splits = {
        "test":  idx[:n_test],
        "val":   idx[n_test:n_test + n_val],
        "train": idx[n_test + n_val:],
    }

    def make_loader(indices, split):
        sub = df.iloc[indices].reset_index(drop=True)
        ds  = OpenIDataset(sub, split=split)
        labels = np.stack(sub["labels"].values)
        workers = 2
        if os.name == "nt":
            workers = 0
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=(split == "train"), num_workers=workers)
        return loader, torch.tensor(labels)

    train_loader, train_labels = make_loader(splits["train"], "train")
    val_loader,   val_labels   = make_loader(splits["val"],   "val")
    test_loader,  test_labels  = make_loader(splits["test"],  "test")

    model     = build_model(args.mode, args.fusion).to(device)
    n_params  = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)

    run_name  = args.mode if args.mode != "fusion" else f"fusion_{args.fusion}"
    ckpt_path = os.path.join(config.MODELS_DIR, f"multimodal_{run_name}.pt")
    stopper   = EarlyStopping(mode="max", checkpoint_path=ckpt_path)

    setup_mlflow(config.EXPERIMENT_MULTIMODAL)
    params = {
        "mode": args.mode, "fusion": args.fusion if args.mode == "fusion" else None,
        "batch_size": args.batch_size, "lr": args.lr, "epochs": args.epochs,
        "text_model": config.TEXT_MODEL_NAME, "n_params": n_params,
    }

    with MLflowRun(run_name=run_name, params=params) as run_ctx:
        run_ctx.log_config(params)

        for epoch in range(args.epochs):
            model.train()
            t0, losses = time.time(), []
            offset = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for image, input_ids, attn in pbar:
                bs = image.size(0)
                labels = train_labels[offset:offset + bs].to(device)
                offset += bs

                optimizer.zero_grad()
                logits = forward_batch(model, args.mode, image, input_ids, attn, device)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = float(np.mean(losses))

            val_logits  = evaluate(model, val_loader, args.mode, criterion, device)
            val_metrics = compute_metrics(val_labels.numpy(), val_logits)
            epoch_time  = time.time() - t0
            print(f"  train_loss={train_loss:.4f}  val_auc={val_metrics['auc_macro']:.4f}  ({epoch_time:.0f}s)")

            log_epoch(epoch, train_loss, val_metrics.get("map_macro", 0.0), val_metrics)
            if stopper(val_metrics["auc_macro"], model):
                print("Early stopping triggered.")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_logits  = evaluate(model, test_loader, args.mode, criterion, device)
        test_metrics = compute_metrics(test_labels.numpy(), test_logits)
        print(f"\nTest AUC (macro): {test_metrics['auc_macro']:.4f}")

        run_ctx.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        run_ctx.log_best_checkpoint(ckpt_path)
        run_ctx.log_model(model, "model")

    print(f"\nDone. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="fusion", choices=["image", "text", "fusion"])
    parser.add_argument("--fusion",     default="late",   choices=["early", "intermediate", "late"])
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    run(parser.parse_args())
