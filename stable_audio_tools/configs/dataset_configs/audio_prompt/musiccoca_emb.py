from pathlib import Path


def get_custom_metadata(info, audio):
    latent_dir = Path(
        "/mnt/gestalt/home/ndrw1221/lab-projects/new-anime-dataset-preprocess/musiccoca_emb"
    )
    rel_path = Path(info["relpath"])
    latent_path = latent_dir / rel_path.with_suffix(".npy")

    return {"audio_prompt": str(latent_path)}
