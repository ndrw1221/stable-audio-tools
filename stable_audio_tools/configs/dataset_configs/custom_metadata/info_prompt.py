def get_custom_metadata(info, audio):

    return {
        "prompt": info["prompt"],
        "seconds_start": info["seconds_start"],
        "seconds_total": info["seconds_total"],
    }
