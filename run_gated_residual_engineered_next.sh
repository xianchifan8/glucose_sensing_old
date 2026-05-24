#!/usr/bin/env bash

# Next experiments for the current best non-MAE path:
# spectrum + full engineered aux + late gated_residual.
# These commands intentionally do not use mae_window_loss.

mkdir -p logs

# 1) Most recommended: keep the successful structure, lower LR/WD slightly
# for a more conservative gated-residual fit.
nohup python main.py --data_source db --db_user Tao_db --model TCN --mode window --window_size 50 --data_fusion --fusion_stage late --fusion_method_late gated_residual --aux_feature_mode engineered --split_strategy random --no_val --train_split 0.8 --test_split 0.2 --normalize --epochs 50 --lr 0.0007 --dropout 0.15 --batch_size 32 --no_wandb --weight_decay 0.0005 --loss Huber --huber_delta 0.8 > logs/Tao_db_random_gres_eng_lr7e4_wd5e4.log 2>&1 &

# 2) Lower dropout: useful if the gated-residual engineered model is now
# slightly underfitting after the aux features became more informative.
nohup python main.py --data_source db --db_user Tao_db --model TCN --mode window --window_size 50 --data_fusion --fusion_stage late --fusion_method_late gated_residual --aux_feature_mode engineered --split_strategy random --no_val --train_split 0.8 --test_split 0.2 --normalize --epochs 50 --lr 0.0007 --dropout 0.10 --batch_size 32 --no_wandb --weight_decay 0.0005 --loss Huber --huber_delta 0.8 > logs/Tao_db_random_gres_eng_lr7e4_dr10_wd5e4.log 2>&1 &

# 3) Stronger Huber quadratic region: may improve peak/valley fitting,
# RMSE, and correlation while preserving robustness.
nohup python main.py --data_source db --db_user Tao_db --model TCN --mode window --window_size 50 --data_fusion --fusion_stage late --fusion_method_late gated_residual --aux_feature_mode engineered --split_strategy random --no_val --train_split 0.8 --test_split 0.2 --normalize --epochs 50 --lr 0.0007 --dropout 0.15 --batch_size 32 --no_wandb --weight_decay 0.0005 --loss Huber --huber_delta 1.0 > logs/Tao_db_random_gres_eng_lr7e4_wd5e4_huber1p0.log 2>&1 &

# 4) More robust Huber: useful if engineered aux still injects occasional
# noisy corrections on motion/contact-quality segments.
nohup python main.py --data_source db --db_user Tao_db --model TCN --mode window --window_size 50 --data_fusion --fusion_stage late --fusion_method_late gated_residual --aux_feature_mode engineered --split_strategy random --no_val --train_split 0.8 --test_split 0.2 --normalize --epochs 50 --lr 0.0007 --dropout 0.15 --batch_size 32 --no_wandb --weight_decay 0.0005 --loss Huber --huber_delta 0.6 > logs/Tao_db_random_gres_eng_lr7e4_wd5e4_huber0p6.log 2>&1 &

# 5) Validate the best current recipe with date-random repeated splits.
# Run this after the random-split sweep identifies the best setting.
# nohup python main.py --data_source db --db_user Tao_db --model TCN --mode window --window_size 50 --data_fusion --fusion_stage late --fusion_method_late gated_residual --aux_feature_mode engineered --split_strategy date_random --date_random_repeats 5 --date_train_days 4 --no_val --train_split 0.8 --test_split 0.2 --normalize --epochs 50 --lr 0.0007 --dropout 0.15 --batch_size 32 --no_wandb --weight_decay 0.0005 --loss Huber --huber_delta 0.8 > logs/Tao_db_daterandom_gres_eng_lr7e4_wd5e4.log 2>&1 &
