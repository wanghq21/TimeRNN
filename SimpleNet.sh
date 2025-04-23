export CUDA_VISIBLE_DEVICES=0


# channel_function=RNN
# temporal_function=patch
# model_name=TSMixer 
# model_name=Transformer2
# model_name=Pathformer
# model_name=TimeMixer
model_name=iTransformer
# model_name=DLinear
# model_name=PatchTST
# model_name=MICN 
# model_name=Transformer2
# model_name=SimpleNet
# model_name=VIT
# model_name=MultiDecomp
# model_name=TSMixer3
# model_name=Channel_conv

# model_name=SimpleTSF



# e_layers=2
# d_model=512
# d_ff=2048
# # npatch=5
# for len in     96  
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/electricity \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --train_epochs 15 \
#     --batch_size 32 \
#     --itr 1 \
#     --use_norm 1
# done


# for len in   12 24 48  96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS03.npz \
#     --model_id PEMS03_96_12 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len $len \
#     --patch   1 24 \
#     --e_layers 4 \
#     --down_sampling_layers 1 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --use_norm 0
# done

# for len in 12 24 48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS04.npz \
#     --model_id PEMS04_96_12 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len $len \
#     --patch  1 24 \
#     --e_layers 3 \
#     --down_sampling_layers 5 \
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --d_ff 1024 \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --use_norm 0
# done


for len in   12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --patch  1 4 12  24 \
    --e_layers 2 \
    --down_sampling_layers 5 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --use_norm 0
done


# for len in   12 24 48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS08.npz \
#     --model_id PEMS08_96_96 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len $len \
#     --patch   1  24 \
#     --e_layers 4 \
#     --down_sampling_layers 5 \
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --use_norm 0
# done


# for len in  96 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Wind \
#     --data_path Wind_50000.csv \
#     --model_id Wind_96_96 \
#     --model $model_name \
#     --data Nodate \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --d_model 128 \
#     --d_ff 128 \
#     --learning_rate 0.001 \
#     --enc_in 255 \
#     --dec_in 255 \
#     --c_out 255 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 128 \
#     --itr 1 \
#     --use_norm 0 
# done

# for len in   720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Solar \
#     --data_path solar_AL.txt \
#     --model_id Solar_96_96 \
#     --model $model_name \
#     --data Solar \
#     --features M \
#     --seq_len 96 \
#     --patch 1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 1 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --itr 1 \
#     --use_norm 0
# done
# python3 -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_mask_0.125 \
#   --mask_rate 0.5 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 0 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 128 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --learning_rate 0.001


# python3 -u run.py \
#   --task_name imputation \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_mask_0.125 \
#   --mask_rate 0.5 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --batch_size 32 \
#   --d_model 512 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 5 \
#   --learning_rate 0.005


# 336 720 1440 2160 2880 3600
#  # weather 
# for len in  96  720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/weather \
#     --data_path weather.csv \
#     --model_id weather_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq t \
#     --seq_len 96 \
#     --patch 1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --itr 1
# done

# for len in    96   720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/electricity \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.0005 \
#     --d_model 512 \
#     --d_core 128 \
#     --d_ff 512 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --itr 1
# done


# for len in   96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_96_96 \
#     --model $model_name \
#     --data ETTm1 \
#     --features M \
#     --freq t \
#     --seq_len 96 \
#     --patch 1  4 12 24  \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.3 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 2048 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# for len in    96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small \
#     --data_path ETTm2.csv \
#     --model_id ETTm2_96_96 \
#     --model $model_name \
#     --data ETTm2 \
#     --features M \
#     --freq t \
#     --seq_len 96  \
#     --patch  1  4  \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 2048 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done


# for len in   720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/electricity \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 4 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch  1 4 8 12 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --itr 1
# done

# for len in  96 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Solar \
#     --data_path Solar_50000.csv \
#     --model_id Solar_96_96 \
#     --model $model_name \
#     --data Nodate \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --d_model 512 \
#     --d_ff 512 \
#     --learning_rate 0.001 \
#     --enc_in 260 \
#     --dec_in 260 \
#     --c_out 260 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1
# done

# for len in  96 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Wind \
#     --data_path Wind_50000.csv \
#     --model_id Wind_96_96 \
#     --model $model_name \
#     --data Nodate \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --d_model 128 \
#     --d_ff 128 \
#     --learning_rate 0.001 \
#     --enc_in 255 \
#     --dec_in 255 \
#     --c_out 255 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1
# done

# for len in  3
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ECL \
#     --data_path electricity.npy \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data npy \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --d_model 512 \
#     --d_ff 512 \
#     --learning_rate 0.001 \
#     --enc_in 370 \
#     --dec_in 370 \
#     --c_out 370 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 20 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1
# done



# for len in 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/traffic \
#     --data_path traffic.csv \
#     --model_id traffic_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
#     --d_model 512 \
#     --d_ff 2048 \
#     --learning_rate 0.001 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 5 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1
# done

# for len in   720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Solar \
#     --data_path solar_AL.txt \
#     --model_id Solar_96_96 \
#     --model $model_name \
#     --data Solar \
#     --features M \
#     --seq_len 96 \
#     --patch 1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 5 \
#     --use_norm 0
# done



# for len in 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/weather \
#     --data_path weather.csv \
#     --model_id weather_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq t \
#     --seq_len 96 \
#     --patch  1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 3 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 2048 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --itr 1
# done


# for len in  720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/electricity \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --patch   1 4 12  24 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 3 \
#     --learning_rate 0.001 \
#     --d_model 256 \
#     --d_core 128 \
#     --d_ff 512 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 3 \
#     --batch_size 32 \
#     --itr 1
# done


