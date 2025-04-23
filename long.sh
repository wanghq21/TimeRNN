
export CUDA_VISIBLE_DEVICES=0


# channel_function=RNN
# temporal_function=patch
# model_name=TSMixer 
# model_name=Transformer2
model_name=Pathformer
model_name=ModernTCN
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
model_name=CycleNet
model_name=Channel_conv


e_layers=2
d_model=512
d_ff=512



for len in        192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --train_epochs 15 \
    --batch_size 32 \
    --itr 1 \
    --use_norm 1
done


for len in 96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 15 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --use_norm 1
done


for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done



# weather数据集将所有的-9999都替换成前二十个数据的平均值

for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather \
    --data_path weather_new.csv \
    --model_id weather_new_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq t \
    --seq_len 96 \
    --patch  1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --down_sampling_layers 3 \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 1
done


# from paper
# Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting
# 和论文Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks
# 其中PEMSD7的形状是 12672 228 npy文件和csv文件是一样的




for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ECL \
    --data_path electricity.npy \
    --model_id ECL2_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 370 \
    --dec_in 370 \
    --c_out 370 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 1
done


for len in   96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ECL \
    --data_path traffic.npy \
    --model_id traffic2_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --patch 1 4 12 24 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 963 \
    --dec_in 963 \
    --c_out 963 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 1
done


# wind 和 Solar数据集来自于
# wind来自于https://www.kaggle.com/datasets/sohier/30-years-of-european-wind-generation/data?select=EMHIRESPV_TSh_CF_Country_19862015.csv
# solar来自于https://www.kaggle.com/datasets/sohier/30-years-of-european-solar-generation


for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path Solar_29.csv \
    --model_id Solar2_96_96 \
    --model $model_name \
    --data Nodate \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 29 \
    --dec_in 29 \
    --c_out 29 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 10 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done

for len in  96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path Solar_260.csv \
    --model_id Solar2_96_96 \
    --model $model_name \
    --data Nodate \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 260 \
    --dec_in 260 \
    --c_out 260 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 10 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done



for len in   96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Wind \
    --data_path Wind_29.csv \
    --model_id Wind_96_96 \
    --model $model_name \
    --data Nodate \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --patch 1 4 12 24 \
    --dropout 0.1 \
    --down_sampling_layers 5 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 29 \
    --dec_in 29 \
    --c_out 29 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 10 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


# # 一直nan
# for len in   96 192 336 720
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
    # --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --patch 1 4 12 24 \
#     --dropout 0.1 \
#     --down_sampling_layers 5 \
    # --d_model $d_model \
    # --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --enc_in 255 \
#     --dec_in 255 \
#     --c_out 255 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --use_norm 0
# done

