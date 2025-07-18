import json
import os


def get_custom_metadata(info, audio):
    metadata_path = (
        "/home/ndrw1221/lab-projects/qwen_caption/pili_no_vocals_47s_qwen.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rel_path = info["relpath"]
    prompt = next(
        (item["caption"] for item in metadata if item["path"] == rel_path), None
    )

    # Use relative path as the prompt
    return {"prompt": prompt}
