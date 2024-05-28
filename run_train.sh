python train.py --dataset_path data/pdnc/novels \
    --save_dir models_storage/cluster/mse_loss \
    --model_state_file model.pth \
    --latent_dimension 1024 \
    --decoder_hidden 4096 \
    --saved_anchor_num 3 \
    --detach_mems_step 12 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --seed 201456 \
    --max_epochs 30 \
    --temperature 0.7