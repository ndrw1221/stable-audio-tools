# Full Fine-tune README

# 訓練資料

請將下載下來的訓練資料解壓縮，並放在與 stable_audio_tools 同層、名為的 dataset 的資料夾中。

正確檔案結構應如下：

```text
stable-audio-tools/
├── ...
├── docs/
├── stable_audio_tools/
└── dataset/
    └── anime-dataset/
        ├── audio/
        ├── pre_encoded/
        ├── anime_dataset.json
        └── anime_metadata.json
```

# 環境設置

## Requirements

- Linux Machine
- GPU with VRAM >= 30G
- CUDA (Development done in CUDA 12.6)
- Python (Development done in Python 3.10)
- PyTorch >= 2.5 for Flash Attention and Flex Attention support

## Install

```bash
$ pip install .
```

# 執行程式

## A. 僅微調 DiT

```bash
$ ./train_dit_only.sh
```

## B. 同時微調 DiT + VAE Decoder

```bash
$ ./train_dit+vae.sh
```

## C. 微調 VAE Decoder 再微調 DiT

1. 微調 VAE

    ```bash
    $ ./train_vae.sh
    ```

2. 將微調好的 VAE 模型 unwrap
   
   ```bash
    $ python3 ./unwrap_model.py \
    --model-config stable-audio-open-1.0/vae/vae_model_config.json \
    --ckpt-path /path/to/wrapped/ckpt \
    --name vae_unwrap
    ```

    請將 `path/to/wrapped/ckpt` 替換成微調好的 vae 的模型 ckpt 檔。

3. 微調 DiT

    ```bash
    $ ./train_dit_with_finetuned_vae.sh
    ```

## D. 模型推論

1. 將微調好 Stable Audio Open 的模型 unwrap（雲端硬碟中的模型檔已經 unwrap 過，可跳過此步驟）

    ```bash
    $ python3 ./unwrap_model.py \
    --model-config stable-audio-open-1.0/vae/vae_model_config.json \
    --ckpt-path /path/to/wrapped/ckpt \
    --name model_unwrap
    ```

    請將 `path/to/wrapped/ckpt` 替換成微調好的 Stable Audio Open 的模型 ckpt 檔。

2. 執行程式開啟推論介面

    ```bash
    $ python run_gradio.py \
    --model-config stable-audio-open-1.0/model_config.json \
    --ckpt-path model_unwrap.ckpt \
    ```