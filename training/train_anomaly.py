import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset import get_chest_mnist_loaders
from models.autoencoder import ConvAE, VAE
from evaluation.metrics import plot_reconstruction
from mlflow_utils.tracking import setup_mlflow, MLflowRun
from training.utils import set_seed, get_device, EarlyStopping, count_parameters
from sklearn.metrics import roc_auc_score


def filter_normal(images, labels):
    mask = labels.sum(dim=1) == 0
    return images[mask]


@torch.no_grad()
def compute_scores(model, loader, device, is_vae):
    model.eval()
    scores, is_abnormal = [], []
    for images, labels in loader:
        images = images.to(device)
        if is_vae:
            x_hat, _, _ = model(images)
        else:
            x_hat, _ = model(images)
        s = model.anomaly_score(images, x_hat).cpu().numpy()
        scores.append(s)
        is_abnormal.append((labels.sum(dim=1) > 0).numpy().astype(int))
    return np.concatenate(scores), np.concatenate(is_abnormal)


def train(args):
    set_seed()
    device = get_device()
    is_vae = args.model == "vae"
    use_amp = not getattr(args, "no_amp", False) and torch.cuda.is_available()
    num_workers = getattr(args, "num_workers", 4)
    print(f"Device: {device} | model: {args.model} | AMP: {use_amp}")
    if use_amp:
        torch.backends.cudnn.benchmark = True

    loaders = get_chest_mnist_loaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
        augment_train=False,
    )

    if is_vae:
        model = VAE(image_size=args.image_size).to(device)
    else:
        model = ConvAE(image_size=args.image_size).to(device)
    n_params = count_parameters(model)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    scaler    = GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(config.MODELS_DIR, f"anomaly_{args.model}.pt")
    stopper   = EarlyStopping(mode="min", checkpoint_path=ckpt_path)

    setup_mlflow(config.EXPERIMENT_ANOMALY)
    params = {
        "model": args.model, "image_size": args.image_size,
        "latent_dim": config.LATENT_DIM, "batch_size": args.batch_size,
        "lr": args.lr, "epochs": args.epochs,
        "beta": config.VAE_BETA if is_vae else None, "n_params": n_params,
    }

    with MLflowRun(run_name=args.model, params=params) as run:
        run.log_config(params)

        for epoch in range(args.epochs):
            model.train()
            t0, losses, recon_losses, kl_losses = time.time(), [], [], []
            pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epochs}")
            for images, labels in pbar:
                normal = filter_normal(images, labels)
                if normal.shape[0] == 0:
                    continue
                normal = normal.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_amp):
                    if is_vae:
                        x_hat, mu, log_var = model(normal)
                        loss, recon, kl = model.elbo_loss(normal, x_hat, mu, log_var)
                        recon_losses.append(recon.item())
                        kl_losses.append(kl.item())
                    else:
                        x_hat, _ = model(normal)
                        loss = torch.nn.functional.mse_loss(x_hat, normal)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = float(np.mean(losses))
            epoch_time = time.time() - t0

            metrics = {"train_loss": train_loss, "epoch_time_s": epoch_time}
            if is_vae:
                metrics["recon_loss"] = float(np.mean(recon_losses))
                metrics["kl_loss"]    = float(np.mean(kl_losses))
            run.log_metrics(metrics, step=epoch)
            print(f"  train_loss={train_loss:.4f}  ({epoch_time:.0f}s)")

            if stopper(train_loss, model):
                print("Early stopping triggered.")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        train_scores, train_flags = compute_scores(model, loaders["train"], device, is_vae)
        normal_scores = train_scores[train_flags == 0]
        threshold = np.percentile(normal_scores, config.ANOMALY_THRESHOLD_PERCENTILE)
        print(f"Anomaly threshold ({config.ANOMALY_THRESHOLD_PERCENTILE}th pct): {threshold:.5f}")

        test_scores, test_flags = compute_scores(model, loaders["test"], device, is_vae)
        if test_flags.sum() > 0 and (test_flags == 0).sum() > 0:
            auc = roc_auc_score(test_flags, test_scores)
        else:
            auc = float("nan")
        print(f"Anomaly detection AUC (normal vs pathological): {auc:.4f}")

        run.log_metrics({
            "anomaly_threshold": float(threshold),
            "anomaly_auc": float(auc),
            "mean_normal_score": float(normal_scores.mean()),
        })

        sample_images, _ = next(iter(loaders["test"]))
        sample_images = sample_images[:8].to(device)
        with torch.no_grad():
            recon = model(sample_images)[0]
        run.log_figure(
            plot_reconstruction(sample_images, recon),
            f"reconstruction_{args.model}.png",
        )
        run.log_best_checkpoint(ckpt_path)
        run.log_model(model, "model")

        np.save(os.path.join(config.MODELS_DIR, f"anomaly_{args.model}_threshold.npy"),
                np.array([threshold]))

    print(f"\nDone. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="vae", choices=["ae", "vae"])
    parser.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--image_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--no_amp",      action="store_true")
    train(parser.parse_args())
