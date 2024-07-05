sigma_list=(1 0.8 0.5 0.3 0)
decoder_hidden=(1024 2048 5000)
feature_dim=(2048 5000 10000)


# freedom loop
for(( a=0; a<5; a++)); do
    # feature dim loop
    for(( b=0; b<3; b++)); do
        python train.py --dataset_path data/pdnc/novels \
            --save_dir models_storage/cluster/l2_epsilon \
            --model_state_file model.pth \
            --decoder_hidden ${decoder_hidden[b]} \
            --feature_dimension ${feature_dim[b]} \
            --feature_freedom 1 \
            --detach_mems_step 5 \
            --learning_rate 2e-6 \
            --weight_decay 0.0 \
            --seed 201456 \
            --max_epochs 15 \
            --book_train_iter 2 \
            --loss_sig ${sigma_list[a]}
    done
done