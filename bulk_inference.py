#!/usr/bin/env python3
"""
Bulk inference script for stable-audio-tools

This script allows you to generate multiple audio samples from text prompts
using the stable-audio-tools diffusion models.

Usage:
    python bulk_inference.py --prompts prompts.txt --output-dir outputs/
    python bulk_inference.py --prompts prompts.txt --model-config model_config.json --ckpt-path model.ckpt
    python bulk_inference.py --prompts prompts.txt --pretrained-name stabilityai/stable-audio-open-1.0
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond


def load_model(
    model_config=None,
    model_ckpt_path=None,
    pretrained_name=None,
    pretransform_ckpt_path=None,
    device="cuda",
    model_half=False,
):
    """Load a model from either pretrained name or config + checkpoint."""

    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
    else:
        raise ValueError(
            "Must specify either pretrained_name or both model_config and model_ckpt_path"
        )

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print(f"Done loading model")
    return model, model_config


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from a text file, one prompt per line."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Convert text to a safe filename."""
    # Remove invalid characters and replace with underscores
    import re

    safe_text = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe_text = re.sub(r"\s+", "_", safe_text)  # Replace spaces with underscores
    safe_text = safe_text.strip("_")  # Remove leading/trailing underscores

    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]

    return safe_text


def generate_audio_batch(
    model, model_config, prompts: List[str], output_dir: str, args
) -> None:
    """Generate audio for a batch of prompts."""

    device = next(model.parameters()).device
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt}")

        # Prepare conditioning
        conditioning = [
            {
                "prompt": prompt,
                "seconds_start": args.seconds_start,
                "seconds_total": args.seconds_total,
            }
        ]

        # Prepare negative conditioning if provided
        negative_conditioning = None
        if args.negative_prompt:
            negative_conditioning = [
                {
                    "prompt": args.negative_prompt,
                    "seconds_start": args.seconds_start,
                    "seconds_total": args.seconds_total,
                }
            ]

        # Set seed for reproducible generation
        seed = args.seed if args.seed != -1 else None
        if args.seed == -1:
            import numpy as np

            seed = np.random.randint(0, 2**32 - 1)

        try:
            # Generate audio
            start_time = time.time()

            if model_type == "diffusion_cond":
                audio = generate_diffusion_cond(
                    model=model,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    batch_size=1,
                    sample_size=sample_size,
                    seed=seed,
                    device=device,
                    sampler_type=args.sampler_type,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    rho=args.rho,
                )
            else:
                raise ValueError(
                    f"Model type {model_type} not supported for bulk inference"
                )

            generation_time = time.time() - start_time

            # Process audio output
            audio = rearrange(audio, "b d n -> d (b n)")
            audio = (
                audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1)
            )

            # Cut to requested duration if specified
            if args.cut_to_seconds_total:
                audio = audio[:, : args.seconds_total * sample_rate]

            # Create output filename
            safe_prompt = sanitize_filename(prompt, max_length=50)
            if args.include_seed_in_filename:
                filename = f"{i+1:03d}_{safe_prompt}_seed{seed}.wav"
            else:
                filename = f"{i+1:03d}_{safe_prompt}.wav"

            output_path = os.path.join(output_dir, filename)

            # Save audio as 16-bit WAV
            audio_int16 = audio.mul(32767).to(torch.int16).cpu()
            torchaudio.save(output_path, audio_int16, sample_rate)

            print(f"✓ Saved: {filename} (took {generation_time:.1f}s)")

            # Save metadata if requested
            if args.save_metadata:
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "seed": seed,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "sampler_type": args.sampler_type,
                    "sigma_min": args.sigma_min,
                    "sigma_max": args.sigma_max,
                    "rho": args.rho,
                    "seconds_start": args.seconds_start,
                    "seconds_total": args.seconds_total,
                    "generation_time": generation_time,
                    "sample_rate": sample_rate,
                    "model_type": model_type,
                }

                metadata_path = output_path.replace(".wav", "_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"✗ Error generating audio for prompt '{prompt}': {e}")
            continue

        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Bulk audio generation using stable-audio-tools"
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

    # Input/output arguments
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated audio files",
    )

    # Generation parameters
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--sampler-type",
        type=str,
        default="dpmpp-3m-sde",
        choices=[
            "dpmpp-3m-sde",
            "dpmpp-2m-sde",
            "k-heun",
            "k-lms",
            "k-dpmpp-2s-a",
            "k-dpm-2",
            "k-euler",
            "k-euler-a",
        ],
        help="Sampler type to use",
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.03, help="Minimum noise level"
    )
    parser.add_argument(
        "--sigma-max", type=float, default=1000, help="Maximum noise level"
    )
    parser.add_argument(
        "--rho", type=float, default=1.0, help="Rho parameter for some samplers"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1 for random)"
    )

    # Audio parameters
    parser.add_argument(
        "--seconds-start", type=int, default=0, help="Start time in seconds"
    )
    parser.add_argument(
        "--seconds-total", type=int, default=30, help="Total duration in seconds"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Negative prompt to avoid certain characteristics",
    )
    parser.add_argument(
        "--cut-to-seconds-total",
        action="store_true",
        help="Cut generated audio to exact duration specified",
    )

    # Output options
    parser.add_argument(
        "--include-seed-in-filename",
        action="store_true",
        help="Include seed in output filename",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save generation metadata as JSON files",
    )

    # Device options
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model_config and not args.ckpt_path:
        parser.error("--ckpt-path is required when using --model-config")

    if not os.path.exists(args.prompts):
        parser.error(f"Prompts file not found: {args.prompts}")

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    print("Loading model...")
    if args.pretrained_name:
        model, model_config = load_model(
            pretrained_name=args.pretrained_name,
            pretransform_ckpt_path=args.pretransform_ckpt_path,
            device=args.device,
            model_half=args.model_half,
        )
    else:
        with open(args.model_config) as f:
            model_config = json.load(f)

        model, model_config = load_model(
            model_config=model_config,
            model_ckpt_path=args.ckpt_path,
            pretransform_ckpt_path=args.pretransform_ckpt_path,
            device=args.device,
            model_half=args.model_half,
        )

    # Load prompts
    print(f"Loading prompts from {args.prompts}...")
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    if len(prompts) == 0:
        print("No prompts found in file!")
        return

    # Print generation settings
    print(f"\nGeneration settings:")
    print(f"  Model type: {model_config['model_type']}")
    print(f"  Sample rate: {model_config['sample_rate']} Hz")
    print(f"  Sample size: {model_config['sample_size']} samples")
    print(f"  Duration: {args.seconds_total} seconds")
    print(f"  Steps: {args.steps}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Sampler: {args.sampler_type}")
    print(f"  Device: {args.device}")
    print(f"  Output dir: {args.output_dir}")

    # Generate audio
    print(f"\nStarting generation for {len(prompts)} prompts...")
    start_total = time.time()

    generate_audio_batch(model, model_config, prompts, args.output_dir, args)

    total_time = time.time() - start_total
    print(
        f"\n✓ Completed! Total time: {total_time:.1f}s ({total_time/len(prompts):.1f}s per prompt)"
    )
    print(f"Generated audio files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
