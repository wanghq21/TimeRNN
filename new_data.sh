
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
model_name=PatchTST
# model_name=Autoformer
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
# model_name=SimpleTSF

e_layers=2
d_model=512
d_ff=512



for len in    96
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
    --dropout 0.1 \
    --learning_rate 0.001  \
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


# for len in  720
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
#     --learning_rate 0.005 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 2 \
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
#     --learning_rate 0.0001 \
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



# # weather数据集将所有的-9999都替换成前二十个数据的平均值
# # 但是换了之后一直nan不知道为什么

# for len in  96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/weather \
#     --data_path weather_new.csv \
#     --model_id weather_new_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq t \
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
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 1
# done


# # from paper
# # Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting
# # 和论文Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks
# # 其中PEMSD7的形状是 12672 228 npy文件和csv文件是一样的




# for len in  96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ECL \
#     --data_path electricity.npy \
#     --model_id ECL2_96_96 \
#     --model $model_name \
#     --data npy \
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
#     --enc_in 370 \
#     --dec_in 370 \
#     --c_out 370 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 1
# done


# for len in   96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ECL \
#     --data_path traffic.npy \
#     --model_id traffic2_96_96 \
#     --model $model_name \
#     --data npy \
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
#     --enc_in 963 \
#     --dec_in 963 \
#     --c_out 963 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 1
# done


# # wind 和 Solar数据集来自于
# # wind来自于https://www.kaggle.com/datasets/sohier/30-years-of-european-wind-generation/data?select=EMHIRESPV_TSh_CF_Country_19862015.csv
# # solar来自于https://www.kaggle.com/datasets/sohier/30-years-of-european-solar-generation


# for len in  96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Solar \
#     --data_path Solar_29.csv \
#     --model_id Solar2_96_96 \
#     --model $model_name \
#     --data Nodate \
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
#     --enc_in 29 \
#     --dec_in 29 \
#     --c_out 29 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done

# for len in  96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Solar \
#     --data_path Solar_260.csv \
#     --model_id Solar2_96_96 \
#     --model $model_name \
#     --data Nodate \
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
#     --enc_in 260 \
#     --dec_in 260 \
#     --c_out 260 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done



# for len in   96 192 336 720
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/Wind \
#     --data_path Wind_29.csv \
#     --model_id Wind_96_96 \
#     --model $model_name \
#     --data Nodate \
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
#     --enc_in 29 \
#     --dec_in 29 \
#     --c_out 29 \
#     --des 'Exp' \
#     --lradj type1 \
#     --train_epochs 10 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done


# # # 一直nan
# # for len in   96 192 336 720
# # do
# #   python3 -u run.py \
# #     --task_name long_term_forecast \
# #     --is_training 1 \
# #     --root_path ./dataset/Wind \
# #     --data_path Wind_50000.csv \
# #     --model_id Wind_96_96 \
# #     --model $model_name \
# #     --data Nodate \
# #     --features M \
# #     --freq h \
# #     --seq_len 96 \
# #     --label_len 96 \
# #     --pred_len $len \
#     # --e_layers $e_layers \
# #     --d_layers 1 \
# #     --factor 3 \
# #     --dropout 0.1 \
#     # --d_model $d_model \
#     # --d_ff $d_ff \
# #     --learning_rate 0.001 \
# #     --enc_in 255 \
# #     --dec_in 255 \
# #     --c_out 255 \
# #     --des 'Exp' \
# #     --lradj type1 \
# #     --train_epochs 10 \
# #     --patience 5 \
# #     --batch_size 32 \
# #     --itr 1 \
# #     --use_norm 0
# # done




# # from paper
# # Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting
# # 和论文Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks
# # 其中PEMSD7的形状是 12672 228 npy文件和csv文件是一样的


# # PEMSD7
# for len in  12 24 48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ECL \
#     --data_path pems.npy \
#     --model_id PEMSD7_96_96 \
#     --model $model_name \
#     --data npy \
#     --features M \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.1 \
#     --d2 0.1 \
#     --learning_rate 0.001 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 228 \
#     --dec_in 228 \
#     --c_out 228 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done




# # pems-bay 和 METR-LA数据集来自于论文
# # Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting


# for len in   12 24  48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS \
#     --data_path PEMS-BAY.csv \
#     --model_id PEMS-BAY_96_96 \
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
#     --enc_in 325 \
#     --dec_in 325 \
#     --c_out 325 \
#     --des 'Exp' \
#     --lradj type1 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done


# for len in   12 24 48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/METR \
#     --data_path METR-LA.csv \
#     --model_id METR-LA_96_96 \
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
#     --enc_in 207 \
#     --dec_in 207 \
#     --c_out 207 \
#     --des 'Exp' \
#     --lradj type1 \
#     --patience 5 \
#     --batch_size 32 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done


# pems 03 04 07 08来自于iTransformer中的介绍


model_name=SOFTS
model_name=SimpleTSF

# # e_layers=4
# for len in  12   24 48  96
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
#     --e_layers $e_layers \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 10 \
#     --use_norm 0
# done


# # e_layers=3
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
#     --e_layers $e_layers \
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 10 \
#     --use_norm 0
# done


# e_layers=3
for len in   96
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
    --train_epochs 1 \
    --use_norm 0
done

model_name=SimpleTSF
e_layers=4
for len in    96
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




