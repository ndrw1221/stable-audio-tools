python3 ./pre_encode.py \
--model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_open_1.0_vae.json \
--ckpt-path stable-audio-open-1.0/vae/vae_model.ckpt \
--dataset-config stable_audio_tools/configs/dataset_configs/new_anime/audio_only.json \
--output-path /home/ndrw1221/lab-projects/new-anime-dataset-preprocess/pre-encoded \
--sample-size 8820000 \
--num-workers 24 \
--batch-size 1 \