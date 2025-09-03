python3 train.py \
--dataset-config stable_audio_tools/configs/dataset_configs/anime_dataset_vae.json \
--model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_open_1.0_vae.json \
--pretrained-ckpt-path stable-audio-open-1.0/vae/vae_model.ckpt \
--save-dir output \
--name vae-anime \
--checkpoint-every 5000 \
--batch-size 8 \
