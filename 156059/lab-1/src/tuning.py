"""Hyperparameter optimization and pruning infrastructure for BloodMNIST.

Provides:
- OptunaPruningCallback: Native PyTorch Lightning callback for Optuna trial pruning.
- BloodObjective: Optuna objective factory supporting configurable parameter search spaces,
  data pipeline configurations (grayscale, augmentations, class weights), and early stopping.
- create_study & run_study: Utilities for creating and executing SQLite-persisted Optuna studies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
import optuna
import torch

from src.data import BloodDataModule, resolve_default_data_path
from src.model import BloodClassifier

logger = logging.getLogger(__name__)


class OptunaPruningCallback(Callback):
    """PyTorch Lightning callback reporting validation metrics to Optuna for trial pruning.

    Subclasses lightning.pytorch.Callback directly to guarantee seamless compatibility
    across PyTorch Lightning 2.x and Optuna without third-party integration package mismatches.

    Args:
        trial: The active Optuna trial.
        monitor: Metric name to evaluate against the pruning rule (e.g. 'val_f1_macro').
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_f1_macro") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        val = (
            current_score.item()
            if isinstance(current_score, torch.Tensor)
            else float(current_score)
        )
        epoch = trainer.current_epoch
        self.trial.report(val, step=epoch)

        if self.trial.should_prune():
            message = (
                f"Trial {self.trial.number} pruned at epoch {epoch} "
                f"with {self.monitor}={val:.4f}"
            )
            logger.info(message)
            raise optuna.TrialPruned(message)


class BloodObjective:
    """Callable Optuna objective for BloodMNIST classification.

    Args:
        data_path: Path to bloodmnist.npz archive.
        grayscale: Whether to convert images to 1-channel grayscale.
        use_augmentations: Whether to apply data augmentations to training set.
        augmentation_type: Augmentation strategy ('none', 'flips_only', 'full').
        use_class_weights: Whether to apply inverse-frequency class weights to loss.
        tune_data_pipeline: Whether Optuna should search over augmentation & class weight options.
        max_epochs: Maximum epochs per trial.
        monitor_metric: Metric monitored for pruning and trial ranking.
        enable_pruning: Whether to enable trial pruning.
        seed: Random seed for reproducibility across trials.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        grayscale: bool = False,
        use_augmentations: bool = True,
        augmentation_type: str = "full",
        use_class_weights: bool = False,
        tune_data_pipeline: bool = True,
        max_epochs: int = 15,
        monitor_metric: str = "val_f1_macro",
        enable_pruning: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_path = Path(data_path) if data_path else resolve_default_data_path()
        self.grayscale = grayscale
        self.use_augmentations = use_augmentations
        self.augmentation_type = augmentation_type
        self.use_class_weights = use_class_weights
        self.tune_data_pipeline = tune_data_pipeline
        self.max_epochs = max_epochs
        self.monitor_metric = monitor_metric
        self.enable_pruning = enable_pruning
        self.seed = seed

    def sample_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define and sample hyperparameters from the search space."""
        return {
            "grayscale": trial.suggest_categorical("grayscale", [False, True]),
            "augmentation_type": trial.suggest_categorical(
                "augmentation_type", ["none", "flips_only", "full"]
            ),
            "use_class_weights": trial.suggest_categorical(
                "use_class_weights", [False, True]
            ),
            "base_channels": trial.suggest_categorical("base_channels", [32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.05),
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", ["adamw", "adam", "sgd"]
            ),
            "scheduler_name": trial.suggest_categorical(
                "scheduler_name", ["cosine", "plateau"]
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        }

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute a single trial evaluation."""
        pl.seed_everything(self.seed, workers=True)
        hparams = self.sample_hyperparameters(trial)

        dm = BloodDataModule(
            data_path=self.data_path,
            batch_size=hparams["batch_size"],
            augmentation_type=hparams["augmentation_type"],
            grayscale=hparams["grayscale"],
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        dm.setup()

        weights = dm.class_weights if hparams["use_class_weights"] else None

        model = BloodClassifier(
            in_channels=dm.in_channels,
            num_classes=dm.num_classes,
            base_channels=hparams["base_channels"],
            dropout=hparams["dropout"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            optimizer_name=hparams["optimizer_name"],
            scheduler_name=hparams["scheduler_name"],
            max_epochs=self.max_epochs,
            class_weights=weights,
        )

        callbacks: list[Callback] = []
        if self.enable_pruning:
            callbacks.append(
                OptunaPruningCallback(trial, monitor=self.monitor_metric)
            )

        callbacks.append(
            EarlyStopping(
                monitor=self.monitor_metric,
                mode="max",
                patience=5,
                verbose=False,
            )
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        try:
            trainer.fit(model, datamodule=dm)
        except optuna.TrialPruned:
            raise

        score = trainer.callback_metrics.get(self.monitor_metric)
        if score is None:
            raise RuntimeError(
                f"Metric '{self.monitor_metric}' was not found in logged trainer metrics."
            )

        return float(score.item() if isinstance(score, torch.Tensor) else score)


def create_study(
    study_name: str = "bloodmnist_hpo",
    storage: str | None = "sqlite:///optuna_study.db",
    direction: str = "maximize",
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Create or load a persistent Optuna study."""
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def run_study(
    objective: BloodObjective,
    study_name: str = "bloodmnist_hpo",
    storage: str | None = "sqlite:///optuna_study.db",
    n_trials: int = 25,
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Create and execute an Optuna optimization study."""
    study = create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)
    return study

