#!/usr/bin/env python3
"""
Inference script for stable-audio-tools
"""

import argparse
import json
import os
import re
import subprocess
import torch
import torchaudio
import numpy as np
from pathlib import Path

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond


def condense_prompt(prompt):
    """Clean up prompt for use in filenames"""
    pattern = r'[\\/:*?"<>|]'
    # Replace special characters with hyphens
    prompt = re.sub(pattern, "-", prompt)
    # set a character limit
    prompt = prompt[:150]
    # zero length prompts may lead to filenames (ie ".wav") which seem cause problems
    if len(prompt) == 0:
        prompt = "_"
    return prompt


def validate_conditioning_defaults(model_config, conditioning_defaults):
    """Validate that all required conditioning parameters are present in conditioning defaults"""
    if (
        "conditioning" not in model_config
        or "configs" not in model_config["conditioning"]
    ):
        print("Warning: No conditioning configuration found in model config")
        return

    required_ids = []
    for config in model_config["conditioning"]["configs"]:
        if "id" in config:
            required_ids.append(config["id"])

    missing_ids = []
    for required_id in required_ids:
        if required_id not in conditioning_defaults and required_id != "prompt":
            # "prompt" can be provided per-sample, so it's not required in defaults
            missing_ids.append(required_id)

    if missing_ids:
        raise ValueError(
            f"Missing required conditioning parameters in conditioning_defaults: {missing_ids}. "
            f"Required parameters from model config: {required_ids}"
        )

    print(f"Conditioning validation passed. Required parameters: {required_ids}")


def save_metadata(metadata, output_path):
    """Save generation metadata as JSON file"""
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")


def convert_audio_format(input_path, output_path, file_format):
    """Convert audio to specified format using ffmpeg"""
    cmd = ""
    if file_format == "m4a aac_he_v2 32k":
        cmd = f'ffmpeg -i "{input_path}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y "{output_path}"'
    elif file_format == "m4a aac_he_v2 64k":
        cmd = f'ffmpeg -i "{input_path}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y "{output_path}"'
    elif file_format == "flac":
        cmd = f'ffmpeg -i "{input_path}" -y "{output_path}"'
    elif file_format == "mp3 320k":
        cmd = f'ffmpeg -i "{input_path}" -b:a 320k -y "{output_path}"'
    elif file_format == "mp3 128k":
        cmd = f'ffmpeg -i "{input_path}" -b:a 128k -y "{output_path}"'
    elif file_format == "mp3 v0":
        cmd = f'ffmpeg -i "{input_path}"  -q:a 0 -y "{output_path}"'
    elif file_format == "wav":
        return input_path  # No conversion needed
    else:
        print(f"Unsupported file format: {file_format}. Saving as WAV.")
        return input_path  # Default to wav

    if cmd:
        cmd += " -loglevel error"  # make output less verbose
        subprocess.run(cmd, shell=True, check=True)
        return output_path
    return input_path


def generate_audio_sample(
    model,
    model_config,
    sample_config,
    generation_config,
    conditioning_defaults,
    output_config,
):
    """Generate a single audio sample"""

    # Merge conditioning with defaults
    conditioning = conditioning_defaults.copy()
    conditioning.update(sample_config)

    # Extract certain conditioning parameters
    prompt = conditioning.get("prompt", "")
    audio_prompt = conditioning.get("audio_prompt", None)
    negative_prompt = conditioning.get("negative_prompt", None)
    seconds_total = conditioning.get("seconds_total", 30)

    # Create conditioning dict - include all conditioning parameters except negative_prompt
    conditioning_dict = {
        k: v for k, v in conditioning.items() if k != "negative_prompt"
    }

    # Create negative conditioning (same as positive but with negative prompt)
    negative_conditioning = None
    if negative_prompt:
        negative_cond_dict = conditioning_dict.copy()
        negative_cond_dict["prompt"] = negative_prompt
        negative_conditioning = [negative_cond_dict]

    # Calculate sample size
    sample_rate = model_config["sample_rate"]
    # sample_size = int(seconds_total * sample_rate)
    sample_size = model_config["sample_size"]

    # Handle seed - generate random seed if -1 was specified
    seed = generation_config.get("seed", -1)
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1)
        generation_config["seed"] = seed

    generation_config["cfg_interval"] = (0, 1)
    generation_config["scale_phi"] = 0

    # Generate audio
    audio = generate_diffusion_cond(
        model=model,
        conditioning=[conditioning_dict],
        negative_conditioning=negative_conditioning,
        sample_size=sample_size,
        **generation_config,
        device=next(model.parameters()).device,
    )

    # Process audio
    audio = audio.squeeze(0)  # Remove batch dimension
    audio = (
        audio.to(torch.float32)
        .div(torch.max(torch.abs(audio)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )

    # Generate filename
    output_dir = Path(output_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    file_naming = output_config.get("file_naming", "prompt")
    file_format = output_config.get("file_format", "wav")

    # Create base filename
    if file_naming == "verbose":
        prompt_condensed = condense_prompt(prompt)
        cfg_scale = generation_config.get("cfg_scale", 6.0)
        if audio_prompt:
            audio_prompt_filename = Path(audio_prompt).stem
            prompt_condensed += f".ap-{audio_prompt_filename}"
        if negative_prompt:
            prompt_condensed += f".neg-{condense_prompt(negative_prompt)}"
        base_name = f"{prompt_condensed}.cfg{cfg_scale}.{seed}"
    elif file_naming == "prompt":
        base_name = condense_prompt(prompt)
    elif file_naming == "audio_prompt" and audio_prompt:
        audio_prompt_filename = Path(audio_prompt).stem
        base_name = audio_prompt_filename
    else:
        base_name = "output"

    # Save as WAV first
    wav_path = output_dir / f"{base_name}.wav"
    torchaudio.save(wav_path, audio, sample_rate)

    # Convert to final format if needed
    if file_format != "wav":
        file_ext = file_format.split(" ")[0].lower()
        final_path = output_dir / f"{base_name}.{file_ext}"
        convert_audio_format(wav_path, final_path, file_format)

        # Remove intermediate WAV file if conversion successful
        if final_path.exists() and final_path != wav_path:
            wav_path.unlink()
            final_output_path = final_path
        else:
            final_output_path = wav_path
    else:
        final_output_path = wav_path

    # Cut to seconds_total if requested
    if output_config.get("cut_to_seconds_total", False) and seconds_total:
        cut_samples = int(seconds_total * sample_rate)
        if audio.shape[-1] > cut_samples:
            audio_cut = audio[..., :cut_samples]
            torchaudio.save(final_output_path, audio_cut, sample_rate)

    # Save metadata if requested
    if output_config.get("save_metadata", False):

        metadata = {
            "generation_config": generation_config,
            "conditioning": conditioning,
            "sample_rate": sample_rate,
            "file_format": file_format,
            "model_config": {
                "sample_rate": model_config["sample_rate"],
                "sample_size": model_config["sample_size"],
                "model_type": model_config["model_type"],
            },
        }
        save_metadata(metadata, final_output_path)

    print(f"Generated: {final_output_path}")
    return final_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Audio generation using stable-audio-tools"
    )

    # Model loading arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--pretrained-name",
        type=str,
        help="Name of pretrained model (e.g., stabilityai/stable-audio-open-1.0)",
    )
    model_group.add_argument(
        "--model-config", type=str, help="Path to model config JSON file"
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Path to model checkpoint (required if using --model-config)",
    )
    parser.add_argument(
        "--pretransform-ckpt-path",
        type=str,
        help="Optional path to pretransform checkpoint",
    )
    parser.add_argument(
        "--model-half",
        action="store_true",
        help="Use half precision (fp16) for model weights",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    # Generation config (required)
    parser.add_argument(
        "--generation-config",
        type=str,
        required=True,
        help="Path to generation configuration JSON file",
    )

    args = parser.parse_args()

    # Load generation config
    with open(args.generation_config, "r") as f:
        config = json.load(f)

    generation_config = config["generation"]
    conditioning_defaults = config["conditioning_defaults"]
    input_config = config["input"]
    output_config = config["output"]

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.pretrained_name:
        print(f"Loading pretrained model: {args.pretrained_name}")
        model, model_config = get_pretrained_model(args.pretrained_name)
    else:
        print(f"Loading model from config: {args.model_config}")
        with open(args.model_config, "r") as f:
            model_config = json.load(f)

        model = create_model_from_config(model_config)

        if args.ckpt_path:
            print(f"Loading checkpoint: {args.ckpt_path}")
            copy_state_dict(model, load_ckpt_state_dict(args.ckpt_path))

    # Load pretransform if specified
    if args.pretransform_ckpt_path:
        print(f"Loading pretransform checkpoint: {args.pretransform_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(args.pretransform_ckpt_path), strict=False
        )

    # Move model to device and set to eval mode
    model.to(device).eval().requires_grad_(False)

    if args.model_half:
        model.to(torch.float16)
        print("Using half precision")

    # Validate conditioning defaults against model config
    validate_conditioning_defaults(model_config, conditioning_defaults)

    # Process input samples
    if input_config["type"] == "file":
        # Load prompts from file
        prompts_file = input_config["prompts_file"]
        print(f"Loading prompts from: {prompts_file}")

        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        samples = [{"prompt": prompt} for prompt in prompts]

    elif input_config["type"] == "config":
        # Use samples from config
        samples = input_config["samples"]
    else:
        raise ValueError(f"Unsupported input type: {input_config['type']}")

    print(f"Generating {len(samples)} samples...")

    # Generate samples
    batch_size = generation_config.get("batch_size", 1)

    for i, sample_config in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}")

        try:
            # Generate single sample or batch
            if batch_size == 1:
                generate_audio_sample(
                    model=model,
                    model_config=model_config,
                    sample_config=sample_config,
                    generation_config=generation_config,
                    conditioning_defaults=conditioning_defaults,
                    output_config=output_config,
                )
            else:
                # For batch generation, repeat the sample
                batch_samples = [sample_config] * batch_size
                for j, batch_sample in enumerate(batch_samples):
                    print(f"  Batch item {j+1}/{batch_size}")
                    generate_audio_sample(
                        model=model,
                        model_config=model_config,
                        sample_config=batch_sample,
                        generation_config=generation_config,
                        conditioning_defaults=conditioning_defaults,
                        output_config=output_config,
                    )

        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            continue

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
