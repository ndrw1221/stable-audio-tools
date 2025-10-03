# Audio Inference

This document describes how to use the inference script to generate audio with stable-audio-tools.

## Overview

The inference script (`inference.py`) provides a flexible way to generate audio samples using pretrained or custom-trained models. It supports various conditioning modes including text prompts, audio prompts, and combined conditioning.

## Basic Usage

```bash
python inference.py --pretrained-name stabilityai/stable-audio-open-1.0 --generation-config path/to/config.json
```

Or with a custom model:

```bash
python inference.py --model-config path/to/model_config.json --ckpt-path path/to/checkpoint.ckpt --generation-config path/to/config.json
```

## Command Line Arguments

### Model Loading Arguments

**Required (choose one):**
- `--pretrained-name`: Name of pretrained model (e.g., `stabilityai/stable-audio-open-1.0`)
- `--model-config`: Path to model config JSON file

**Optional:**
- `--ckpt-path`: Path to model checkpoint (required when using `--model-config`)
- `--pretransform-ckpt-path`: Optional path to pretransform checkpoint
- `--model-half`: Use half precision (fp16) for model weights
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

### Generation Configuration

**Required:**
- `--generation-config`: Path to generation configuration JSON file

## Generation Configuration File Structure

The generation configuration file is a JSON file that defines all aspects of the generation process. It consists of four main sections:

### 1. Generation Parameters

The `generation` section controls the diffusion sampling process:

```json
{
    "generation": {
        "steps": 100,
        "cfg_scale": 7.0,
        "batch_size": 1,
        "sampler_type": "dpmpp-3m-sde",
        "sigma_min": 0.01,
        "sigma_max": 100,
        "rho": 1.0,
        "seed": -1
    }
}
```

**Parameters:**
- `steps`: Number of diffusion steps (more steps = higher quality, slower generation)
- `cfg_scale`: Classifier-free guidance scale (higher = more prompt adherence)
- `batch_size`: Number of samples to generate per prompt
- `sampler_type`: Diffusion sampler algorithm
- `sigma_min`/`sigma_max`: Noise schedule parameters
- `rho`: Noise schedule parameter
- `seed`: Random seed (-1 for random seed)

### 2. Conditioning Defaults

The `conditioning_defaults` section provides default values for conditioning parameters:

```json
{
    "conditioning_defaults": {
        "prompt": "",
        "audio_prompt": null,
        "seconds_start": 0,
        "seconds_total": 30,
        "negative_prompt": null
    }
}
```

**Parameters:**
- `prompt`: Default text prompt
- `audio_prompt`: Path to audio file for audio conditioning (`.npy` embedding files)
- `seconds_start`: Starting position in generated audio
- `seconds_total`: Total duration of generated audio
- `negative_prompt`: Negative prompt for guidance

### 3. Input Configuration

The `input` section defines how prompts/samples are provided. Two modes are supported:

#### Config Mode (Inline Samples)

```json
{
    "input": {
        "type": "config",
        "samples": [
            {
                "prompt": "A jazz piece with saxophone"
            },
            {
                "prompt": "Electronic dance music",
                "audio_prompt": "/path/to/audio_embedding.npy"
            }
        ]
    }
}
```

#### File Mode (External Prompts File)

```json
{
    "input": {
        "type": "file",
        "prompts_file": "path/to/prompts.txt"
    }
}
```

The prompts file should contain one prompt per line:
```
Cyberpunk
Jazz music with piano
Electronic beats
```

### 4. Output Configuration

The `output` section controls how generated audio is saved:

```json
{
    "output": {
        "output_dir": "generated/outputs",
        "file_format": "mp3 v0",
        "file_naming": "verbose",
        "save_metadata": true,
        "cut_to_seconds_total": true
    }
}
```

**Parameters:**
- `output_dir`: Directory to save generated files
- `file_format`: Output audio format (see supported formats below)
- `file_naming`: Filename generation strategy
- `save_metadata`: Whether to save generation metadata as JSON
- `cut_to_seconds_total`: Whether to trim audio to `seconds_total` duration

#### Supported File Formats

- `wav`: Uncompressed WAV (default)
- `mp3 v0`: MP3 variable bitrate (highest quality)
- `mp3 320k`: MP3 320kbps
- `mp3 128k`: MP3 128kbps
- `flac`: FLAC lossless compression
- `m4a aac_he_v2 32k`: M4A with HE-AAC v2 at 32kbps
- `m4a aac_he_v2 64k`: M4A with HE-AAC v2 at 64kbps

#### File Naming Strategies

- `prompt`: Use the prompt text as filename
- `verbose`: Include prompt, CFG scale, seed, and conditioning info
- `audio_prompt`: Use audio prompt filename (when audio conditioning is used)

## Example Configuration Files

### Basic Text-to-Audio Generation

```json
{
    "generation": {
        "steps": 100,
        "cfg_scale": 7.0,
        "batch_size": 1,
        "sampler_type": "dpmpp-3m-sde",
        "sigma_min": 0.01,
        "sigma_max": 100,
        "rho": 1.0,
        "seed": -1
    },
    "conditioning_defaults": {
        "seconds_start": 0,
        "seconds_total": 30,
        "negative_prompt": "low quality, distorted"
    },
    "input": {
        "type": "config",
        "samples": [
            {
                "prompt": "A relaxing jazz piece with piano and saxophone"
            },
            {
                "prompt": "Upbeat electronic dance music with heavy bass"
            }
        ]
    },
    "output": {
        "output_dir": "generated/basic_examples",
        "file_format": "mp3 v0",
        "file_naming": "prompt",
        "save_metadata": true,
        "cut_to_seconds_total": true
    }
}
```

### Audio + Text Conditioning

```json
{
    "generation": {
        "steps": 100,
        "cfg_scale": 7.0,
        "batch_size": 1,
        "sampler_type": "dpmpp-3m-sde",
        "sigma_min": 0.01,
        "sigma_max": 100,
        "rho": 1.0,
        "seed": -1
    },
    "conditioning_defaults": {
        "prompt": "",
        "audio_prompt": null,
        "seconds_start": 90,
        "seconds_total": 180,
        "negative_prompt": null
    },
    "input": {
        "type": "config",
        "samples": [
            {
                "audio_prompt": "/path/to/reference_audio.npy"
            },
            {
                "audio_prompt": "/path/to/reference_audio.npy",
                "prompt": "A jazz piece with smooth saxophone"
            }
        ]
    },
    "output": {
        "output_dir": "generated/audio_conditioning",
        "file_format": "wav",
        "file_naming": "verbose",
        "save_metadata": true,
        "cut_to_seconds_total": true
    }
}
```

### Batch Generation from File

```json
{
    "generation": {
        "steps": 100,
        "cfg_scale": 6.0,
        "batch_size": 1,
        "sampler_type": "dpmpp-3m-sde",
        "sigma_min": 0.01,
        "sigma_max": 100,
        "rho": 1.0,
        "seed": -1
    },
    "conditioning_defaults": {
        "seconds_start": 0,
        "seconds_total": 40,
        "negative_prompt": "low quality, distorted"
    },
    "input": {
        "type": "file",
        "prompts_file": "prompts.txt"
    },
    "output": {
        "output_dir": "generated/from_file",
        "file_format": "mp3 v0",
        "file_naming": "verbose",
        "save_metadata": true,
        "cut_to_seconds_total": true
    }
}
```

## Audio Conditioning

For audio conditioning, you need to provide pre-computed audio embeddings as `.npy` files. These embeddings are typically generated using a separate audio encoder that matches your model's conditioning architecture.

The `audio_prompt` parameter should point to a NumPy file containing the audio embedding:

```json
{
    "audio_prompt": "/path/to/audio_embedding.npy"
}
```

## Metadata Output

When `save_metadata` is enabled, a JSON file is saved alongside each generated audio file containing:

- Generation configuration used
- Conditioning parameters
- Model information
- Sample rate and format details

Example metadata structure:
```json
{
    "generation_config": {
        "steps": 100,
        "cfg_scale": 7.0,
        "seed": 12345
    },
    "conditioning": {
        "prompt": "A jazz piece",
        "seconds_total": 30
    },
    "sample_rate": 44100,
    "file_format": "mp3 v0",
    "model_config": {
        "sample_rate": 44100,
        "sample_size": 1048576,
        "model_type": "diffusion_cond"
    }
}
```

## Tips and Best Practices

### Generation Quality
- Use higher `steps` (100-200) for better quality
- Adjust `cfg_scale` (6-8) for prompt adherence vs. diversity balance
- Use `negative_prompt` to avoid unwanted characteristics

### Performance
- Use `--model-half` for faster inference with slightly reduced quality
- Lower `steps` for faster generation
- Use appropriate `batch_size` based on GPU memory

### Audio Duration
- `seconds_total` determines the target duration
- Longer durations may require more GPU memory
- Use `cut_to_seconds_total` to ensure exact duration

### File Management
- Use descriptive `output_dir` names for organization
- Enable `save_metadata` for reproducibility
- Choose appropriate `file_format` based on quality/size requirements

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size`, use `--model-half`, or generate shorter audio
2. **Poor Quality**: Increase `steps`, adjust `cfg_scale`, or improve prompts
3. **Model Loading Errors**: Ensure checkpoint paths are correct and model config matches
4. **Conditioning Validation Errors**: Check that all required conditioning parameters are provided in `conditioning_defaults`

### FFmpeg Requirements

Audio format conversion requires FFmpeg to be installed:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

For M4A formats with high-efficiency AAC, ensure FFmpeg is compiled with `libfdk_aac` support.