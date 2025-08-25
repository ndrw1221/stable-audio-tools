import json
import random

STYLE_KEYWORD = "Japanese anime"
AUDIO_META_PATH = "/home/ndrw1221/lab-projects/new-anime-dataset-preprocess/new_anime_dataset_with_titles.json"
ANIME_META_PATH = (
    "/home/ndrw1221/lab-projects/new-anime-dataset-preprocess/anime_metadata.json"
)

_cache = {}


def get_custom_metadata(info, audio):
    global _cache
    if not _cache:
        # Load files once
        try:
            with open(AUDIO_META_PATH) as f:
                _cache["audio_meta"] = json.load(f)

            with open(ANIME_META_PATH) as f:
                _cache["anime_meta"] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading metadata files: {e}")

    rel_path = info["relpath"]

    if not rel_path:
        raise ValueError("Relative path not found in info.")

    audio_meta = _cache["audio_meta"]
    anime_meta = _cache["anime_meta"]

    audio_meta_entry = next(
        (entry for entry in audio_meta if entry["path"] == rel_path), None
    )

    if audio_meta_entry is None:
        raise ValueError(f"No metadata found for path: {rel_path}")

    tags = audio_meta_entry.get("tags", "")
    caption = audio_meta_entry.get("caption", "")
    anime_title = audio_meta_entry.get("anime_title", "")

    if not tags or not caption:
        raise ValueError("Missing tags or caption in audio metadata.")

    # Use caption or tags for prompt, 50/50 chance
    use_caption = random.random() < 0.5
    prompt = caption if use_caption else tags

    if not anime_title or anime_title not in anime_meta:
        prompt = f"{STYLE_KEYWORD}. {prompt}"
        return {"prompt": prompt}

    # Always add the english title
    anime_meta_entry = anime_meta[anime_title]
    en_title = anime_meta_entry["en_title"]
    prompt = f"{en_title}. {prompt}"

    # Randomly add additional anime metadata only when using tags (not caption)
    if not use_caption:
        for meta_key in anime_meta_entry:
            if meta_key == "en_title" or meta_key == "tw_title":
                continue

            meta_value = anime_meta_entry.get(meta_key, "")
            if meta_value and random.random() < 0.2:
                prompt += f", {meta_value}"

    prompt = f"{STYLE_KEYWORD}. {prompt}"
    return {"prompt": prompt}
