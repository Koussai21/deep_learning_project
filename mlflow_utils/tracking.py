"""
MLflow tracking utilities.

Every training script calls setup_mlflow() first, then uses the context
manager run_context() to wrap the training loop. Artifacts (figures,
best model checkpoint, config YAML) are logged automatically.
"""
import os
import yaml
import mlflow
import mlflow.pytorch
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_mlflow(experiment_name: str) -> None:
    db_path = os.path.join(config.BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)


class MLflowRun:
    """Context manager that wraps an MLflow run and exposes helper methods."""

    def __init__(self, run_name: str, params: dict = None):
        self.run_name = run_name
        self.params   = params or {}

    def __enter__(self):
        self._run = mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self.params)
        return self

    def __exit__(self, *_):
        mlflow.end_run()

    # ── Logging helpers ───────────────────────────────────────────────────

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_figure(self, fig, filename: str) -> None:
        tmp = os.path.join(config.BASE_DIR, filename)
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp)
        os.remove(tmp)

    def log_model(self, model: torch.nn.Module, name: str) -> None:
        mlflow.pytorch.log_model(model, name)

    def log_best_checkpoint(self, path: str) -> None:
        mlflow.log_artifact(path, artifact_path="checkpoints")

    def log_config(self, cfg: dict) -> None:
        tmp = os.path.join(config.BASE_DIR, "run_config.yaml")
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)
        mlflow.log_artifact(tmp)
        os.remove(tmp)


def log_epoch(step: int, train_loss: float, val_loss: float, val_metrics: dict) -> None:
    """Convenience function to log a single epoch from inside a training loop."""
    mlflow.log_metrics(
        {"train_loss": train_loss, "val_loss": val_loss, **val_metrics},
        step=step,
    )
