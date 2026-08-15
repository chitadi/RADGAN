import os
import argparse

import yaml
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from loguru import logger

import models
from datamodule import DataModule
from utils import safe_open_yaml, stringify

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = os.path.join(CURRENT_DIR, "config", "train_hifi_gan.yaml")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "results")


def _init_model(model_name, model_params):
    model_cls = getattr(models, model_name)
    return model_cls(**model_params)


def _build_trainer(trainer_cfg, default_root_dir, loggers, callbacks):
    """Create a Trainer using only the fields supplied in trainer_cfg."""
    trainer_kwargs = {
        "default_root_dir": default_root_dir,
        "logger": loggers,
        "callbacks": list(callbacks),
    }
    allowed_keys = {
        "max_epochs",
        "limit_train_batches",
        "limit_val_batches",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "log_every_n_steps",
        "devices",
        "accelerator",
        "strategy",
        "accumulate_grad_batches",
    }
    for key in allowed_keys:
        if key in trainer_cfg:
            trainer_kwargs[key] = trainer_cfg[key]
    return pl.Trainer(**trainer_kwargs)


def _setup_loggers(model_name, run_name, save_dir, logging_cfg):
    """Set up CSV logger and optionally WandB logger."""
    pl_logger = CSVLogger("logs", name=f"{model_name}_{run_name}")
    loggers = [pl_logger]

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = logging_cfg.get("use_wandb", False) and wandb_api_key is not None

    if use_wandb:
        wandb.login(key=wandb_api_key)
        wandb_logger = WandbLogger(
            project=os.environ.get("WANDB_PROJECT", "rad-gan"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"{model_name}_{run_name}",
            save_dir=save_dir,
            log_model=False,
        )
        loggers.append(wandb_logger)

    return loggers


def _validate(trainer, data_module, model_name, best_ckpt, model_params,
              save_dir, config_path, fast_dev_run=True):
    logger.info(f"The best model can be found in {best_ckpt}")
    model_cls = getattr(models, model_name)
    best_model = model_cls.load_from_checkpoint(best_ckpt, strict=True, **model_params)

    logger.info("Running validate with best model..")
    best_model.heavy_eval = True
    val_metrics = trainer.validate(best_model, datamodule=data_module)

    output_yaml = os.path.join(save_dir, "fast_dev_run.yaml" if fast_dev_run else "output.yaml")

    payload = {
        "best_model_path": os.path.abspath(best_ckpt) or None,
        "metrics": val_metrics,
        "config": os.path.abspath(config_path),
    }
    with open(output_yaml, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    logger.info(f"Validation results saved to {output_yaml}")
    logger.info("To prepare a submission package, run:")
    logger.info(f"  python save_for_submission.py -c {os.path.abspath(output_yaml)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAD-GAN")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to training config YAML")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to a Lightning checkpoint to resume from")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run a quick smoke test (1 epoch, 5 batches)")
    args = parser.parse_args()

    seed_everything(8, workers=True)

    config = safe_open_yaml(args.config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_name = config["model"]
    model_params = config["model_params"]
    model_params_str = stringify(model_params, delimiter="_")
    save_dir = os.path.join(OUTPUT_DIR, f"{model_name}_{model_params_str}")
    os.makedirs(save_dir, exist_ok=True)

    logging_cfg = config.get("trainer", {}).get("logging", {})

    model = _init_model(model_name, model_params)
    data_module = DataModule(**config["datamodule"])

    if args.fast_dev_run:
        # --- Fast dev run: smoke test ---
        logger.info("Running fast development run (1 epoch, 5 batches).")
        fast_dev_dir = os.path.join(OUTPUT_DIR, f"{model_name}_{model_params_str}_fast_dev_run")
        os.makedirs(fast_dev_dir, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
            monitor="val/loss",
            filename="{epoch:03d}-{val/loss:.2f}",
            mode="min",
        )
        loggers = _setup_loggers(model_name, f"{model_params_str}_fast_dev_run", fast_dev_dir, logging_cfg)

        fast_trainer_cfg = {**config["trainer"]["fast_dev"], **logging_cfg}
        trainer = _build_trainer(
            fast_trainer_cfg,
            default_root_dir=fast_dev_dir,
            loggers=loggers,
            callbacks=[ckpt_callback],
        )

        trainer.fit(model, data_module)
        _validate(trainer, data_module, model_name,
                  ckpt_callback.best_model_path, model_params,
                  fast_dev_dir, args.config, fast_dev_run=True)

    else:
        # --- Full training ---
        logger.info("Running full training.")
        ckpt_callback = ModelCheckpoint(
            monitor="val/pesq_estoi_mfcc",
            save_top_k=5,
            filename="{epoch:03d}-{step}-{val/loss:.2f}",
            mode="max",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        loggers = _setup_loggers(model_name, model_params_str, save_dir, logging_cfg)

        full_trainer_cfg = {**config["trainer"]["full"], **logging_cfg}
        trainer = _build_trainer(
            full_trainer_cfg,
            default_root_dir=save_dir,
            loggers=loggers,
            callbacks=[ckpt_callback, lr_monitor],
        )

        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
        _validate(trainer, data_module, model_name,
                  ckpt_callback.best_model_path, model_params,
                  save_dir, args.config, fast_dev_run=False)
