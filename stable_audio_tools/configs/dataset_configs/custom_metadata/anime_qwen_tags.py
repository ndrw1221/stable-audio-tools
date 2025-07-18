import json


def get_custom_metadata(info, audio):
    metadata_path = "/home/ndrw1221/lab-projects/anime-dataset-preprocess/anime-dataset/anime_tags_with_titles_japanese_anime.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rel_path = info["relpath"]
    prompt = next(
        (item["caption"] for item in metadata if item["path"] == rel_path), None
    )

    if prompt is None:
        raise ValueError(f"Relative path '{rel_path}' not found in metadata.")

    # Use relative path as the prompt
    return {
        "prompt": prompt,
        "seconds_start": info["seconds_start"],
        "seconds_total": info["seconds_total"],
    }
