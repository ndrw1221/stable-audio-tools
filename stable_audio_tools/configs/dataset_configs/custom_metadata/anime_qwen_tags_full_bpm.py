import json


def get_custom_metadata(info, audio):
    metadata_path = "/home/ndrw1221/lab-projects/anime-dataset-preprocess/anime-dataset/anime_tags_full_bpm.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rel_path = info["relpath"]
    prompt = next(
        (item["caption"] for item in metadata if item["path"] == rel_path), None
    )
    bpm = next((item["bpm"] for item in metadata if item["path"] == rel_path), None)

    if prompt is None:
        raise ValueError(f"Relative path '{rel_path}' not found in metadata.")

    if bpm is None:
        raise ValueError(f"BPM for relative path '{rel_path}' not found in metadata.")

    return {
        "prompt": prompt,
        "bpm": bpm,
    }
