import json


def get_custom_metadata(info, audio):
    metadata_path = "/mnt/gestalt/home/ndrw1221/lab-projects/anime-dataset-preprocess/anime-dataset/anime-dataset.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rel_path = info["relpath"]
    prompt = next(
        (item["caption"] for item in metadata if item["path"] == rel_path), None
    )

    # Use relative path as the prompt
    return {"prompt": prompt}
