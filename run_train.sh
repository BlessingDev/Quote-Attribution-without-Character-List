sigma_list=(1 0.8 0.5 0.3 0)
decoder_hidden=(1024 2048 5000)
feature_dim=(2048 5000 10000)

books=(0 5 9 13 16)

# freedom loop
for(( a=0; a<5; a++)); do
    python train.py --dataset_path data/pdnc/novels \
        --save_dir models_storage/classification/foreach \
        --model_state_file model.pth \
        --decoder_hidden 1024 \
        --feature_dimension 2048 \
        --feature_freedom 1 \
        --detach_mems_step 5 \
        --learning_rate 2e-6 \
        --weight_decay 0.0 \
        --seed 201456 \
        --max_epochs 20 \
        --early_stopping_criteria 20 \
        --book_train_iter 1 \
        --loss_sig 1 \
        --val_books ${books[a]} \
        --train_books ${books[a]}
done