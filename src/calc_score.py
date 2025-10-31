#!/usr/bin/env python3
import pytorch_lightning as pl
from loguru import logger

from datamodule import DataModule
from utils import safe_open_yaml
import models
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/train_dcctn.yaml")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "logs/dcctn__learning_rate=0.0003_weight_decay=1e-05_betas=[0.5, 0.999]_stft_loss_config={'fft_size': 320, 'hop_size': 80, 'win_length': 320, 'scale_invariance': False, 'w_sc': 0.0, 'weight': 50.0}/version_8/checkpoints/epoch=004-step=1655-val/loss=81.22.ckpt")
EVAL_STAGE = "val"  # switch to "test" if you want test_dataloader()

def build_trainer(eval_cfg):
    trainer_kwargs = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "precision": eval_cfg.get("precision", "32-true"),
        "accelerator": eval_cfg.get("accelerator", "auto"),
        "devices": eval_cfg.get("devices", 1),
    }
    if "strategy" in eval_cfg:
        trainer_kwargs["strategy"] = eval_cfg["strategy"]
    return pl.Trainer(**trainer_kwargs)

def main():
    config = safe_open_yaml(CONFIG_PATH)
    data_module = DataModule(**config["datamodule"])
    data_module.setup("fit")

    model_name = config["model"]
    model_cls = getattr(models, model_name)
    model = model_cls.load_from_checkpoint(CHECKPOINT_PATH, **config["model_params"])
    model.heavy_eval = True

    eval_cfg = config.get("trainer", {}).get("eval", {})
    trainer = build_trainer(eval_cfg)

    logger.info(f"Validating {model_name} with checkpoint {CHECKPOINT_PATH}")
    dataloader_method = getattr(data_module, f"{EVAL_STAGE}_dataloader")
    metrics = trainer.validate(model, dataloaders=dataloader_method(), verbose=True)

    logger.info(f"{EVAL_STAGE} metrics: {metrics}")

if __name__ == "__main__":
    main()
