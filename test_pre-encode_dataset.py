# %%
import torch
import json
import os
import pytorch_lightning as pl
from stable_audio_tools.data.dataset import create_dataloader_from_config

torch.multiprocessing.set_sharing_strategy("file_system")
seed = 42

# Set a different seed for each process if using SLURM
if os.environ.get("SLURM_PROCID") is not None:
    seed += int(os.environ.get("SLURM_PROCID"))

pl.seed_everything(seed, workers=True)

# Get JSON config from args.model_config
with open("stable-audio-open-1.0/model_config.json") as f:
    model_config = json.load(f)

with open("latent_tags_full.json") as f:
    dataset_config = json.load(f)

train_dl = create_dataloader_from_config(
    dataset_config,
    batch_size=32,
    num_workers=4,
    sample_rate=model_config["sample_rate"],
    sample_size=model_config["sample_size"],
    audio_channels=model_config.get("audio_channels", 2),
)

# %%
info = train_dl.dataset[0][1]
info

# %%
info["padding_mask"][0].shape
# %%
info["audio"][0].shape
# %%
