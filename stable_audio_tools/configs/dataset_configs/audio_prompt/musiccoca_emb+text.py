import json
import random
from pathlib import Path

_cache = {}


def get_custom_metadata(info, audio):
    latent_dir = Path(
        "/mnt/gestalt/home/ndrw1221/lab-projects/new-anime-dataset-preprocess/musiccoca_emb"
    )
    prompt_dir = Path(
        "/mnt/gestalt/home/ndrw1221/lab-projects/new-anime-dataset-preprocess/new_anime_dataset.json"
    )

    global _cache
    if not _cache:
        try:
            with open(prompt_dir) as f:
                _cache["prompts"] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading prompt file: {e}")

    rel_path = Path(info["relpath"])
    latent_path = latent_dir / rel_path.with_suffix(".npy")

    prompts = _cache["prompts"]
    prompt_entry = next(
        (entry for entry in prompts if entry["path"] == str(rel_path)), None
    )

    if prompt_entry is None:
        raise ValueError(f"No prompt found for path: {rel_path}")

    tags = prompt_entry["tags"]
    caption = prompt_entry["caption"]

    use_caption = random.random() < 0.5
    prompt = caption if use_caption else tags

    return {"audio_prompt": str(latent_path), "prompt": prompt}
