freedom_list=(1 3 5)
decoder_hidden=(1024 2048 5000)
feature_dim=(2048 5000 10000)


# freedom loop
for(( a=0; a<3; a++)); do
    # feature dim loop
    for(( b=0; b<3; b++)); do
        python train.py --dataset_path data/pdnc/novels \
            --save_dir models_storage/cluster/mse_epsilon \
            --model_state_file model.pth \
            --decoder_hidden ${decoder_hidden[b]} \
            --feature_dimension ${feature_dim[b]} \
            --feature_freedom ${freedom_list[a]} \
            --detach_mems_step 5 \
            --learning_rate 2e-6 \
            --weight_decay 0.1 \
            --seed 201456 \
            --max_epochs 15 \
            --book_train_iter 2 \
            --loss_sig 0.5
    done
done