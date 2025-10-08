import torch
import json
import os
import pytorch_lightning as pl

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import (
    copy_state_dict,
    load_ckpt_state_dict,
    remove_weight_norm_from_model,
)
from stable_audio_tools.training import (
    create_training_wrapper_from_config,
    create_demo_callback_from_config,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    # Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False,
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(args.pretransform_ckpt_path)
        )

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    # Determine the run name - use run_name if provided, otherwise fallback to wandb run ID
    run_name = getattr(args, "run_name", "") or None
    project_name = getattr(args, "name", None)

    if args.logger == "wandb":
        # Set wandb run name if run_name is provided
        wandb_kwargs = {"project": project_name}
        if run_name:
            wandb_kwargs["name"] = run_name

        logger = pl.loggers.WandbLogger(**wandb_kwargs)
        logger.watch(training_wrapper)

        if args.save_dir:
            # Use run_name if provided, otherwise use wandb run ID
            folder_name = run_name if run_name else logger.experiment.id
            if isinstance(folder_name, str):
                checkpoint_dir = os.path.join(
                    args.save_dir,
                    project_name,
                    folder_name,
                    "checkpoints",
                )
                # Create the demos directory structure
                demo_dir = os.path.join(
                    args.save_dir, project_name, folder_name, "demos"
                )
                os.makedirs(demo_dir, exist_ok=True)
            else:
                checkpoint_dir = None
                demo_dir = None
        else:
            checkpoint_dir = None
            demo_dir = None
    elif args.logger == "comet":
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir:
            # Use run_name if provided, otherwise use comet version
            folder_name = (
                run_name
                if run_name
                else (logger.version if isinstance(logger.version, str) else None)
            )
            if folder_name:
                checkpoint_dir = os.path.join(
                    args.save_dir, project_name, folder_name, "checkpoints"
                )
                # Create the demos directory structure
                demo_dir = os.path.join(
                    args.save_dir, project_name, folder_name, "demos"
                )
                os.makedirs(demo_dir, exist_ok=True)
            else:
                checkpoint_dir = args.save_dir if args.save_dir else None
                demo_dir = None
        else:
            checkpoint_dir = None
            demo_dir = None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        if args.save_dir and args.name and run_name:
            demo_dir = os.path.join(args.save_dir, args.name, run_name, "demos")
            os.makedirs(demo_dir, exist_ok=True)
        else:
            demo_dir = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(
            model_config, demo_dl=val_dl, demo_dir=demo_dir
        )
    else:
        demo_callback = create_demo_callback_from_config(
            model_config, demo_dl=train_dl, demo_dir=demo_dir
        )

    # Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == "wandb":
        push_wandb_config(logger, args_dict)
    elif args.logger == "comet":
        logger.log_hyperparams(args_dict)

    # Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        else:
            strategy = args.strategy
    else:
        strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"

    val_args = {}

    if args.val_every > 0:
        val_args.update(
            {
                "check_val_every_n_epoch": None,
                "val_check_interval": args.val_every,
            }
        )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[
            ckpt_callback,
            demo_callback,
            exc_callback,
            save_model_config_callback,
        ],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,  # If you need to debug validation, change this line
        **val_args,
    )

    trainer.fit(
        training_wrapper,
        train_dl,
        val_dl,
        ckpt_path=args.ckpt_path if args.ckpt_path else None,
    )


if __name__ == "__main__":
    main()
