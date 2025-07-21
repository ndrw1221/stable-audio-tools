python3 ./pre_encode.py \
--model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_open_1.0_vae.json \
--ckpt-path stable-audio-tools/unwrapped_ckpt/vae-pili-82000-unwrapped.ckpt \
--dataset-config stable_audio_tools/configs/dataset_configs/pili/audio_only.json \
--output-path /mnt/gestalt/home/ndrw1221/datasets/anime-dataset/ft-vae-pre-encode/ \
--sample-size 2097152 \
--num-workers 24 \
--batch-size 2 \