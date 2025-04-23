
export CUDA_VISIBLE_DEVICES=0


# channel_function=RNN
# temporal_function=patch
# model_name=TSMixer 
# model_name=Transformer2
# model_name=Pathformer
# model_name=ModernTCN
# model_name=TimeMixer
# model_name=iTransformer
# model_name=DLinear
# model_name=PatchTST
# model_name=MICN 
# model_name=Transformer2
# model_name=SimpleNet
# model_name=VIT
# model_name=MultiDecomp
# model_name=TSMixer3
# model_name=CycleNet
# model_name=SOFTS
# model_name=Leddam
# model_name=PaiFilter
# model_name=CARD
# model_name=DUET
# model_name=CCM_patchtst
# model_name=CCM_dlinear
# model_name=Fredformer
# model_name=Channel_conv

model_name=SegRNN

# model_name=SimpleTSF

e_layers=2
d_model=512
d_ff=512


seg_len=24

# for len in      192  336 720
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
#     --learning_rate 0.001  \
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


# for len in  96 192 336 720
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
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 15 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --use_norm 1
# done



# for len in   96 192 336 720
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
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 137 \
#     --dec_in 137 \
#     --c_out 137 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done




# for len in   96 192 336  720
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
#     --seq_len 96 \
#     --patch  1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 3 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --itr 1
# done


# # e_layers=2
# # d_model=16
# # d_ff=32


# for len in  96 192  336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --patch 1 4 12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --itr 1
# done





# for len in 96 192 336  720
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
#     --seq_len 96 \
#     --patch 1 4  12 24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 5 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done


# for len in  96 192 336   720
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
#     --seq_len 96  \
#     --patch  1 4 12 24   \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 3 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done



# for len in  96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 96 \
#     --label_len 96 \
#     --patch 1 4 12 24  \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 2 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# for len in  96  192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small \
#     --data_path ETTh2.csv \
#     --model_id ETTh2_96_96 \
#     --model $model_name \
#     --data ETTh2 \
#     --features M \
#     --seq_len 96 \
#     --patch    24 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --down_sampling_layers 1 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done


# e_layers=2
# d_model=128
# # d_ff=256


seg_len=12

# e_layers=4
for len in   12 24 48  96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


# e_layers=3
for len in 12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


# e_layers=3
for len in  12 24 48 96
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
    --e_layers $e_layers \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


# e_layers=4
for len in   12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done





