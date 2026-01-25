
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb

import yaml
import os
from parse import parse
from datetime import datetime

import models
import argparse
import pandas as pd
# from torch.nn import module_list
from datamodule import DataModule
from functools import partial
from loguru import logger
from utils import safe_open_yaml, stringify

current_directory = os.path.dirname(__file__)
python_file_name = os.path.splitext(os.path.basename(__file__))[0]

# OUTPUT_DIR = "/results/" 
OUTPUT_DIR = os.path.join(current_directory, "..", "results")
CONFIG_FILE = os.path.join(current_directory, "config", "train_hifi_gan.yaml")
# CONFIG_FILE = "/src/config/train_wnd_unet.yaml"
# CONFIG_FILE = "/src/config/train_baseline.yaml"
# CONFIG_FILE = "/src/config/train_diffwave_v1.yaml"
# CONFIG_FILE = os.path.join(current_directory, "config", "train_ap_bwe.yaml")

# WandB Configuration
WANDB_PROJECT = "rase-challenge"  # Project name in WandB
WANDB_ENTITY = "team-quazo"    # Replace with your WandB team/organization name
WANDB_API_KEY = "72d48843d3dc81a30bd737bf05e90efff610f190"  # Your API key
wandb.login(key=WANDB_API_KEY)

def _init_model(model_name, model_params):
    model_cls       = getattr(models, model_name)
    model           = model_cls(**model_params)

    return model

def _build_trainer(trainer_cfg: dict, default_root_dir: str, loggers, callbacks):
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


def _validate(trainer, data_module, model_name, best_ckpt, model_params, save_dir, fast_dev_run=True):


    best_ckpt = best_ckpt
    logger.info(f"The best model can be found in {best_ckpt}")
    model_cls       = getattr(models, model_name)
    best_model = model_cls.load_from_checkpoint(best_ckpt, strict=True, **model_params)

    logger.info("Running validate with best model..")
    best_model.heavy_eval = True
    val_metrics = trainer.validate(best_model, datamodule=data_module)
    
    if fast_dev_run:
        output_yaml = f"{save_dir}/fast_dev_run.yaml"
    else:
        output_yaml = f"{save_dir}/output.yaml"
    
    logger.info("If you want to use the model with this validation result as the model for the evalAI test, run the following")
    logger.info(f"python save_for_submission.py -c {os.path.abspath(output_yaml)}.")
    logger.info(f"You don't have to run test as it will be done on our server.")
    payload = {
        "best_model_path": os.path.abspath(best_ckpt) or None,
        "metrics": val_metrics,
        "config": os.path.abspath(CONFIG_FILE)
    }
    with open(output_yaml, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to a Lightning checkpoint to resume from")
    args = parser.parse_args()
    seed_everything(8, workers=True)

    config = safe_open_yaml(CONFIG_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    model_name      = config["model"]
    model_params    = config["model_params"]
    model_params_str = stringify(model_params, delimiter="_")
    save_dir                = f"{OUTPUT_DIR}/{model_name}_{model_params_str}"
    save_dir_fast_dev_run   = f"{OUTPUT_DIR}/{model_name}_{model_params_str}_fast_dev_run"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_fast_dev_run, exist_ok=True)

    model = _init_model(model_name, model_params)

    data_module = DataModule(**config["datamodule"])

    ckpt_callback = ModelCheckpoint(
        monitor="val/loss", 
        filename='{epoch:03d}-{val/loss:.2f}',
        mode="min",
    )
    pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}_fast_dev_run")

    logger.info("Running fast development run with single epoch.")
    
    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{model_name}_{model_params_str}_fast_dev_run",
        save_dir=save_dir_fast_dev_run,
        log_model=False
    ) if WANDB_API_KEY else None
    
    # Use both CSV and WandB loggers
    loggers = [pl_logger]
    if wandb_logger:
        loggers.append(wandb_logger)
    
    # uncomment the following trainer for non-dcctn models
    # trainer = pl.Trainer(
    #     max_epochs=1,
    #     default_root_dir=save_dir_fast_dev_run,
    #     callbacks=[ckpt_callback],
    #     logger=loggers,
    #     gradient_clip_val=0.5,
    #     gradient_clip_algorithm="norm",
    #     limit_train_batches=5,
    #     limit_val_batches=5,
    #     devices=[1]
    #     # track_grad_norm=2
    # )
    logging_cfg = config["trainer"].get("logging", {})
    fast_trainer_cfg = {**config["trainer"]["fast_dev"], **logging_cfg}
    trainer = _build_trainer(
        fast_trainer_cfg,
        default_root_dir=save_dir_fast_dev_run,
        loggers=loggers,
        callbacks=[ckpt_callback],
    )


    # trainer.fit(model, data_module)
    # _validate(trainer, data_module, model_name, ckpt_callback.best_model_path, model_params, save_dir_fast_dev_run, fast_dev_run=True)
    ########## RUN ACTUAL #################
    
    ckpt_callback = ModelCheckpoint(
        # monitor="val/loss", 
        monitor="val/pesq_estoi_mfcc",
        save_top_k=5,
        filename='{epoch:03d}-{step}-{val/loss:.2f}',
        # mode="min",
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}")
    
    # Setup WandB logger for actual training
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{model_name}_{model_params_str}",
        save_dir=save_dir,
        log_model=False
    ) if WANDB_API_KEY else None
    
    # Use both CSV and WandB loggers
    loggers = [pl_logger]
    if wandb_logger:
        loggers.append(wandb_logger)
    
    # uncomment the following trainer for non-dcctn models
    # trainer = pl.Trainer(
    #     max_epochs=1,
    #     default_root_dir=save_dir,
    #     callbacks=[ckpt_callback, lr_monitor],
    #     gradient_clip_val=1.0,
    #     gradient_clip_algorithm="norm",
    #     log_every_n_steps=50,
    #     logger=loggers,
    #     devices=[1]
    #     # track_grad_norm=2
    # )
    logging_cfg = config["trainer"].get("logging", {})
    full_trainer_cfg = {**config["trainer"]["full"], **logging_cfg}
    trainer = _build_trainer(
        full_trainer_cfg,
        default_root_dir=save_dir,
        loggers=loggers,
        callbacks=[ckpt_callback, lr_monitor],
    )

    model = _init_model(model_name, model_params)

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    # best_ckpt = "/home/jagat/Chittem/RASE-Challenge-team-quazo/src/logs/dcctn_learning_rate=0.0001_weight_decay=0.0001_betas=79791a4d_stft_loss_config=multi-7c02bb24/version_0/checkpoints/epoch=019-step=840-val/loss=37.73.ckpt"
    # best_ckpt = "/home/jagat/Chittem/RASE-Challenge-team-quazo/src/logs/dcctn_learning_rate=0.0001_weight_decay=0.0001_betas=79791a4d_stft_loss_config=multi-e743a5b6/version_0/checkpoints/epoch=013-step=588-val/loss=98.89.ckpt"
    # best_ckpt = "/home/jagat/Chittem/RASE-Challenge-team-quazo/src/logs/dcctn_learning_rate=0.0001_weight_decay=0.0001_betas=79791a4d_stft_loss_config=multi-494a0ca3/version_0/checkpoints/epoch=013-step=588-val/loss=88.82.ckpt"
    best_ckpt = "/home/jagat/Chittem/RASE-Challenge-team-quazo/src/logs/hifigan_learning_rate=0.0001/version_7/checkpoints/epoch=041-step=55524-val/loss=3.04.ckpt"
    data_module.setup("fit")
    _validate(trainer, data_module, model_name,
            #   ckpt_callback.best_model_path, 
            best_ckpt,
            model_params, save_dir, fast_dev_run=False)


# import os
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything
# from pytorch_lightning.loggers import CSVLogger


# import yaml
# import os
# from parse import parse
# from datetime import datetime

# import models
# import argparse
# import pandas as pd
# # from torch.nn import module_list
# from datamodule import DataModule
# from functools import partial
# from loguru import logger
# from utils import safe_open_yaml, stringify

# current_directory = os.path.dirname(__file__)
# python_file_name = os.path.splitext(os.path.basename(__file__))[0]

# OUTPUT_DIR = "/results/" 
# CONFIG_FILE = "/src/config/train_hifi_gan.yaml"

# def _init_model(model_name, model_params):
#     model_cls       = getattr(models, model_name)
#     model           = model_cls(**model_params)

#     return model

# def _validate(trainer, data_module, model_name, best_ckpt, model_params, save_dir, fast_dev_run=True):


#     best_ckpt = ckpt_callback.best_model_path
#     logger.info(f"The best model can be found in {best_ckpt}")
#     model_cls       = getattr(models, model_name)
#     best_model = model_cls.load_from_checkpoint(best_ckpt, **model_params)

#     logger.info("Running validate with best model..")
#     best_model.heavy_eval = True
#     val_metrics = trainer.validate(best_model, datamodule=data_module)
    
#     if fast_dev_run:
#         output_yaml = f"{save_dir}/fast_dev_run.yaml"
#     else:
#         output_yaml = f"{save_dir}/output.yaml"
    
#     logger.info("If you want to use the model with this validation result as the model for the evalAI test, run the following")
#     logger.info(f"python save_for_submission.py -c {os.path.abspath(output_yaml)}.")
#     logger.info(f"You don't have to run test as it will be done on our server.")
#     payload = {
#         "best_model_path": os.path.abspath(best_ckpt) or None,
#         "metrics": val_metrics,
#         "config": os.path.abspath(CONFIG_FILE)
#     }
#     with open(output_yaml, "w") as f:
#         yaml.safe_dump(payload, f, sort_keys=False)


# if __name__ == "__main__":

#     config = safe_open_yaml(CONFIG_FILE)
#     os.makedirs(OUTPUT_DIR, exist_ok=True)


#     model_name      = config["model"]
#     model_params    = config["model_params"]
#     model_params_str = stringify(model_params, delimiter="_")
#     save_dir                = f"{OUTPUT_DIR}/{model_name}_{model_params_str}"
#     save_dir_fast_dev_run   = f"{OUTPUT_DIR}/{model_name}_{model_params_str}_fast_dev_run"
#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(save_dir_fast_dev_run, exist_ok=True)

#     model = _init_model(model_name, model_params)

#     data_module = DataModule(**config["datamodule"])
#     seed_everything(8, workers=True)

#     ckpt_callback = ModelCheckpoint(
#         monitor="val/loss", 
#         filename='{epoch:03d}-{val/loss:.2f}',
#         mode="min",
#     )
#     pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}_fast_dev_run")

#     logger.info("Running fast development run with single epoch.")
#     trainer = pl.Trainer(
#         max_epochs=1,
#         default_root_dir=save_dir_fast_dev_run,
#         callbacks=[ckpt_callback],
#         logger=pl_logger,
#         gradient_clip_val=0.5,
#         limit_train_batches=5,
#         limit_val_batches=5,
#         devices=1
#     )

#     trainer.fit(model, data_module)
#     _validate(trainer, data_module, model_name, ckpt_callback.best_model_path, model_params, save_dir_fast_dev_run, fast_dev_run=True)
#     ########## RUN ACTUAL #################
    
#     ckpt_callback = ModelCheckpoint(
#         monitor="val/loss", 
#         save_top_k=5,
#         filename='{epoch:03d}-{step}-{val/loss:.2f}',
#         mode="min",
#     )
#     pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}")
#     trainer = pl.Trainer(
#         max_epochs=20,
#         default_root_dir=save_dir,
#         callbacks=[ckpt_callback],
#         gradient_clip_val=0.5,
#         logger=pl_logger,
#         devices=1
#     )

#     model = _init_model(model_name, model_params)

#     # trainer.fit(model, data_module)
#     _validate(trainer, data_module, model_name, ckpt_callback.best_model_path, model_params, save_dir, fast_dev_run=False)