export CUDA_VISIBLE_DEVICES=0

model_name=TSMixer

# for rate in  0.375 0.5
# do
#     python3 -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_mask_0.125 \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 0 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --batch_size 32 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --itr 1 \
#     --top_k 3 \
#     --learning_rate 0.005
# done



# for rate in 0.125 0.25 0.375 0.5
# do
#     python -u run.py \
#     --task_name imputation \
#     --is_training 1 \
#     --root_path ./dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_mask_0.125 \
#     --mask_rate $rate \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 0 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --batch_size 16 \
#     --d_model 128 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --itr 1 \
#     --top_k 3 \
#     --lradj type3 \
#     --train_epochs 100 \
#     --patience 10 \
#     --learning_rate 0.001
# done



python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 3 \
  --learning_rate 0.001


