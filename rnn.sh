export CUDA_VISIBLE_DEVICES=0
model_name=Channel_conv
# model_name=iTransformer
# model_name=SMamba
# model_name=SimpleTSF


# e_layers=4
# d_model=512
# d_ff=2048
# # npatch=5
# for len in      192 336  720
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
#     --n_patch 5 \
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

# long_term_forecast_traffic_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm512_nh8_el4_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.3906003534793854, mae:0.25991374254226685, rmse:0.6249802708625793, mape:2.6652958393096924, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic_96_96_Channel_conv_custom_ftM_sl96_ll96_pl192_dm512_nh8_el4_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.41311484575271606, mae:0.2692595422267914, rmse:0.6427401304244995, mape:2.725898027420044, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic_96_96_Channel_conv_custom_ftM_sl96_ll96_pl336_dm512_nh8_el4_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.4317932426929474, mae:0.2778868079185486, rmse:0.6571097373962402, mape:2.7309160232543945, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic_96_96_Channel_conv_custom_ftM_sl96_ll96_pl720_dm512_nh8_el4_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.46434029936790466, mae:0.2968463897705078, rmse:0.6814252138137817, mape:2.8776488304138184, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001



e_layers=2
d_model=512
d_ff=2048
# npatch=5
for len in  96
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
    --n_patch 5 \
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


# long_term_forecast_ECL_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.13672304153442383, mae:0.2318865805864334, rmse:0.36976078152656555, mape:2.4007389545440674, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL_96_96_Channel_conv_custom_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.15543924272060394, mae:0.24890826642513275, rmse:0.3942578434944153, mape:2.6239089965820312, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL_96_96_Channel_conv_custom_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.17030784487724304, mae:0.2651807963848114, rmse:0.41268372535705566, mape:2.682685613632202, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL_96_96_Channel_conv_custom_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.1958836168050766, mae:0.2894996106624603, rmse:0.44258740544319153, mape:2.9192583560943604, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001


# e_layers=2
# d_model=512
# d_ff=2048
# for len in    720
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
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.7 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# npatch=sqrt

# long_term_forecast_ETTm1_96_96_Channel_conv_ETTm1_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.31003087759017944, mae:0.35060885548591614, rmse:0.5568041801452637, mape:2.1892802715301514, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm1_96_96_Channel_conv_ETTm1_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.35748592019081116, mae:0.37636056542396545, rmse:0.5979012846946716, mape:2.246828079223633, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm1_96_96_Channel_conv_ETTm1_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.39239153265953064, mae:0.4000357985496521, rmse:0.6264116168022156, mape:2.3185768127441406, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm1_96_96_Channel_conv_ETTm1_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.45919421315193176, mae:0.437692254781723, rmse:0.6776387095451355, mape:2.5195939540863037, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001



# e_layers=2
# d_model=512
# d_ff=512
# for len in      720
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
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.7 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# npatch=sqrt


# long_term_forecast_ETTm2_96_96_Channel_conv_ETTm2_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.1715407520532608, mae:0.2551252543926239, rmse:0.41417479515075684, mape:1.115541696548462, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm2_96_96_Channel_conv_ETTm2_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.23576262593269348, mae:0.29751792550086975, rmse:0.48555395007133484, mape:1.2431907653808594, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm2_96_96_Channel_conv_ETTm2_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.2945369482040405, mae:0.3354428708553314, rmse:0.5427125692367554, mape:1.3563493490219116, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTm2_96_96_Channel_conv_ETTm2_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.39264386892318726, mae:0.39289090037345886, rmse:0.6266130208969116, mape:1.5547105073928833, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001




# e_layers=2
# d_model=512
# d_ff=2048
# for len in   96 192 336 
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
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.7 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
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
#     --root_path ./dataset/ETT-small \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.9 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# npatch=sqrt

# long_term_forecast_ETTh1_96_96_Channel_conv_ETTh1_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.3785938024520874, mae:0.391528457403183, rmse:0.6152997612953186, mape:9.757454872131348, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh1_96_96_Channel_conv_ETTh1_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.4325468838214874, mae:0.42212364077568054, rmse:0.6576829552650452, mape:9.78891372680664, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh1_96_96_Channel_conv_ETTh1_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.47752881050109863, mae:0.4456872344017029, rmse:0.6910346150398254, mape:9.683247566223145, patch:[1, 4, 12, 24], dropout:0.7, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh1_96_96_Channel_conv_ETTh1_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.48057740926742554, mae:0.47041061520576477, rmse:0.6932368874549866, mape:9.920741081237793, patch:[1, 4, 12, 24], dropout:0.9, d2:0.1, learning_rate:0.0001



# e_layers=2
# d_model=512
# d_ff=512
# for len in   96 192 336
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
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.6 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done
# e_layers=1
# d_model=512
# d_ff=512
# for len in  720
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
#     --freq h \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.6 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1
# done

# npatch=sqrt

# long_term_forecast_ETTh2_96_96_Channel_conv_ETTh2_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.28078287839889526, mae:0.33584827184677124, rmse:0.5298895239830017, mape:1.4229751825332642, patch:[1, 4, 12, 24], dropout:0.6, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh2_96_96_Channel_conv_ETTh2_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.36087679862976074, mae:0.38660699129104614, rmse:0.6007302403450012, mape:1.5637935400009155, patch:[1, 4, 12, 24], dropout:0.6, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh2_96_96_Channel_conv_ETTh2_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.39869946241378784, mae:0.41653987765312195, rmse:0.6314265131950378, mape:1.6678088903427124, patch:[1, 4, 12, 24], dropout:0.6, d2:0.1, learning_rate:0.0001

# long_term_forecast_ETTh2_96_96_Channel_conv_ETTh2_ftM_sl96_ll96_pl720_dm512_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.409549742937088, mae:0.4325198829174042, rmse:0.6399607062339783, mape:1.9479765892028809, patch:[1, 4, 12, 24], dropout:0.6, d2:0.1, learning_rate:0.0001



# e_layers=2
# d_model=512
# d_ff=2048
# npatch=sqrt
# for len in   96 192  336  720
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --freq d \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.9 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 1
# done


# long_term_forecast_Exchange_96_96_Channel_conv_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.08159242570400238, mae:0.19855329394340515, rmse:0.28564387559890747, mape:1.2049957513809204, patch:[1, 4, 12, 24], dropout:0.9, d2:0.1, learning_rate:0.0001

# long_term_forecast_Exchange_96_96_Channel_conv_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.171431764960289, mae:0.2934722900390625, rmse:0.4140431880950928, mape:1.8554538488388062, patch:[1, 4, 12, 24], dropout:0.9, d2:0.1, learning_rate:0.0001

# long_term_forecast_Exchange_96_96_Channel_conv_custom_ftM_sl96_ll48_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.32133713364601135, mae:0.40964704751968384, rmse:0.5668660402297974, mape:2.910726308822632, patch:[1, 4, 12, 24], dropout:0.9, d2:0.1, learning_rate:0.0001

# long_term_forecast_Exchange_96_96_Channel_conv_custom_ftM_sl96_ll48_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.8297469019889832, mae:0.684531569480896, rmse:0.9109044671058655, mape:6.043493270874023, patch:[1, 4, 12, 24], dropout:0.9, d2:0.1, learning_rate:0.0001


# e_layers=2
# d_model=512
# d_ff=2048
# for len in    336  
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --freq t \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.3 \
#     --learning_rate 0.001 \
#     --n_patch -1 \
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
# e_layers=2
# d_model=512
# d_ff=2048
# for len in    720
# do
#     python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --freq t \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.4 \
#     --learning_rate 0.0001 \
#     --n_patch -1 \
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

# npatch=sqrt

# long_term_forecast_weather_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.1542162299156189, mae:0.19801567494869232, rmse:0.39270374178886414, mape:11.897042274475098, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.0001

# long_term_forecast_weather_96_96_Channel_conv_custom_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.2040187418460846, mae:0.24459026753902435, rmse:0.4516843259334564, mape:13.630643844604492, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.0001

# long_term_forecast_weather_96_96_Channel_conv_custom_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.2631833851337433, mae:0.2891565263271332, rmse:0.513014018535614, mape:13.660138130187988, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.0001

# long_term_forecast_weather_96_96_Channel_conv_custom_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.3450391888618469, mae:0.34277212619781494, rmse:0.5874003767967224, mape:12.651398658752441, patch:[1, 4, 12, 24], dropout:0.4, d2:0.1, learning_rate:0.0001



# e_layers=1
# d_model=512
# d_ff=512
# for len in   96 192  336 720
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
#     --n_patch 20 \
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

# npatch=20

# long_term_forecast_Solar_96_96_Channel_conv_Solar_ftM_sl96_ll96_pl96_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.18194423615932465, mae:0.2500474154949188, rmse:0.4265492260456085, mape:2.0524110794067383, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.0001

# long_term_forecast_Solar_96_96_Channel_conv_Solar_ftM_sl96_ll96_pl192_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.19119231402873993, mae:0.2564168870449066, rmse:0.4372554421424866, mape:1.9742544889450073, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.0001

# long_term_forecast_Solar_96_96_Channel_conv_Solar_ftM_sl96_ll96_pl336_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.19608579576015472, mae:0.259987473487854, rmse:0.44281575083732605, mape:2.1119306087493896, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.0001

# long_term_forecast_Solar_96_96_Channel_conv_Solar_ftM_sl96_ll96_pl720_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.20851671695709229, mae:0.26860901713371277, rmse:0.45663630962371826, mape:2.2304937839508057, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.0001





# e_layers=2
# d_model=512
# d_ff=512
# # npatch=5
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
#     --n_patch 5 \
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


# long_term_forecast_ECL2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:202.25146484375, mae:1.020752191543579, rmse:14.221513748168945, mape:inf, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:324.8232116699219, mae:1.1311753988265991, rmse:18.022851943969727, mape:inf, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:528.5955200195312, mae:1.2846790552139282, rmse:22.9912052154541, mape:inf, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_ECL2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:1051.6783447265625, mae:1.5942083597183228, rmse:32.42959213256836, mape:inf, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001







# e_layers=4
# d_model=512
# d_ff=2048
# # npatch=sqrt
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
#     --n_patch 5 \
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


# long_term_forecast_traffic2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl96_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.21700246632099152, mae:0.19307571649551392, rmse:0.46583524346351624, mape:2.115011215209961, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl192_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.24780698120594025, mae:0.2111380249261856, rmse:0.49780213832855225, mape:2.478684902191162, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl336_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.2705889344215393, mae:0.2351214736700058, rmse:0.5201816558837891, mape:2.7843093872070312, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_traffic2_96_96_Channel_conv_npy_ftM_sl96_ll96_pl720_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.2748866379261017, mae:0.25841692090034485, rmse:0.5242963433265686, mape:3.6802685260772705, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001






# e_layers=3
# d_model=512
# d_ff=512
# # npatch=sqrt
# for len in   12 24 48 96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS \
#     --data_path PEMSD7.npy \
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
#     --dropout 0.3 \
#     --learning_rate 0.001 \
#     --n_patch -1 \
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


# long_term_forecast_PEMSD7_96_96_Channel_conv_npy_ftM_sl96_ll96_pl12_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.24596181511878967, mae:0.256637841463089, rmse:0.49594536423683167, mape:2.238586664199829, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMSD7_96_96_Channel_conv_npy_ftM_sl96_ll96_pl24_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.3175089955329895, mae:0.29384106397628784, rmse:0.5634793639183044, mape:2.4713408946990967, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMSD7_96_96_Channel_conv_npy_ftM_sl96_ll96_pl48_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.3762373626232147, mae:0.330311119556427, rmse:0.6133819222450256, mape:2.7541918754577637, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMSD7_96_96_Channel_conv_npy_ftM_sl96_ll96_pl96_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.41438862681388855, mae:0.35223355889320374, rmse:0.6437302231788635, mape:3.017310619354248, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001



# e_layers=3
# d_model=512
# d_ff=512
# # # npatch=sqrt
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
#     --freq t \
#     --features M \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.3 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --n_patch -1 \
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



# long_term_forecast_PEMS-BAY_96_96_Channel_conv_custom_ftM_sl96_ll96_pl12_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.25677481293678284, mae:0.22241343557834625, rmse:0.5067295432090759, mape:2.1048035621643066, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS-BAY_96_96_Channel_conv_custom_ftM_sl96_ll96_pl24_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.33263304829597473, mae:0.2530156970024109, rmse:0.576743483543396, mape:2.2668070793151855, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS-BAY_96_96_Channel_conv_custom_ftM_sl96_ll96_pl48_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.4211385250091553, mae:0.29017964005470276, rmse:0.6489518880844116, mape:2.448308229446411, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS-BAY_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm512_nh8_el3_dl1_df512_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.49594664573669434, mae:0.32378479838371277, rmse:0.704234778881073, mape:2.749779224395752, patch:[1, 4, 12, 24], dropout:0.3, d2:0.1, learning_rate:0.001




# e_layers=1
# d_model=128
# d_ff=128
# # npatch=sqrt
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
#     --freq t \
#     --seq_len 96 \
#     --label_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --d_layers 1 \
#     --factor 3 \
#     --dropout 0.3 \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --n_patch -1 \
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


# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl12_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.4344787299633026, mae:0.3349645435810089, rmse:0.6591500043869019, mape:1.487392544746399, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl24_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.6196236610412598, mae:0.41363394260406494, rmse:0.7871617674827576, mape:1.6756958961486816, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl48_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.9120533466339111, mae:0.5221809148788452, rmse:0.9550148248672485, mape:1.7158615589141846, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:1.0866475105285645, mae:0.6274310350418091, rmse:1.0424238443374634, mape:1.8367536067962646, patch:[1, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl12_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.42810067534446716, mae:0.33103713393211365, rmse:0.6542940139770508, mape:1.507741093635559, patch:[1, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl24_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.6279020309448242, mae:0.43697288632392883, rmse:0.7924026846885681, mape:1.621877908706665, patch:[1, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl48_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.862951397895813, mae:0.5111491084098816, rmse:0.9289517998695374, mape:1.6798642873764038, patch:[1, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:1.1629198789596558, mae:0.6047586798667908, rmse:1.0783876180648804, mape:1.8280895948410034, patch:[1, 24], dropout:0.3, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl12_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.4241824448108673, mae:0.3250783681869507, rmse:0.6512929201126099, mape:1.5216872692108154, patch:[1, 24], dropout:0.5, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl24_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.6089538335800171, mae:0.41221773624420166, rmse:0.780354917049408, mape:1.710226058959961, patch:[1, 24], dropout:0.5, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl48_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:0.8810368180274963, mae:0.5154187679290771, rmse:0.9386355876922607, mape:1.6954783201217651, patch:[1, 24], dropout:0.5, d2:0.1, learning_rate:0.001

# long_term_forecast_METR-LA_96_96_Channel_conv_custom_ftM_sl96_ll96_pl96_dm128_nh8_el1_dl1_df128_expand2_dc4_fc3_ebtimeF_dtFalse_Exp_0  
# mse:1.1096150875091553, mae:0.646034300327301, rmse:1.0533826351165771, mape:1.8051425218582153, patch:[1, 24], dropout:0.5, d2:0.1, learning_rate:0.001






# e_layers=4
# d_model=512
# d_ff=512
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
#     --e_layers $e_layers \
#     --n_patch -1 \
#     --enc_in 358 \
#     --dec_in 358 \
#     --c_out 358 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done


# e_layers=3
# d_model=512
# d_ff=2048
# for len in   96
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
#     --n_patch -1 \
#     --enc_in 307 \
#     --dec_in 307 \
#     --c_out 307 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done

# npatch=sqrt

# long_term_forecast_PEMS03_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl12_dm512_nh8_el4_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.05948644131422043, mae:0.15906846523284912, rmse:0.2438984215259552, mape:1.3409557342529297, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS03_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl24_dm512_nh8_el4_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.07265391200780869, mae:0.17564211785793304, rmse:0.2695438861846924, mape:1.5273278951644897, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS03_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl48_dm512_nh8_el4_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.09594713896512985, mae:0.20322880148887634, rmse:0.3097533583641052, mape:1.7831496000289917, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS03_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl96_dm512_nh8_el4_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.14556121826171875, mae:0.24483485519886017, rmse:0.38152486085891724, mape:2.1375572681427, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS04_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl12_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.06679084151983261, mae:0.16609492897987366, rmse:0.25843924283981323, mape:1.200665831565857, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS04_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl24_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.07529368251562119, mae:0.1773274689912796, rmse:0.27439695596694946, mape:1.2919706106185913, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS04_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl48_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.08833575248718262, mae:0.1928330510854721, rmse:0.2972133159637451, mape:1.4129064083099365, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS04_96_12_Channel_conv_PEMS_ftM_sl96_ll96_pl96_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.10595016181468964, mae:0.21186105906963348, rmse:0.32549986243247986, mape:1.5366356372833252, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001



# e_layers=2
# d_model=512
# d_ff=2048
# for len in     12 24 48  96
# do
#   python3 -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/PEMS/ \
#     --data_path PEMS07.npz \
#     --model_id PEMS07_96_96 \
#     --model $model_name \
#     --data PEMS \
#     --features M \
#     --seq_len 96 \
#     --pred_len $len \
#     --e_layers $e_layers \
#     --n_patch 5 \
#     --enc_in 883 \
#     --dec_in 883 \
#     --c_out 883 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done

# npatch=5

# long_term_forecast_PEMS07_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl12_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.05479727312922478, mae:0.1514640748500824, rmse:0.23408816754817963, mape:1.4917007684707642, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS07_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl24_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.06676191836595535, mae:0.16811245679855347, rmse:0.25838327407836914, mape:1.601432204246521, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS07_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl48_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.08537085354328156, mae:0.19097551703453064, rmse:0.29218292236328125, mape:1.7969365119934082, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS07_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.10638398677110672, mae:0.2129562944173813, rmse:0.3261655867099762, mape:1.971985936164856, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001


# e_layers=2
# d_model=512
# d_ff=512
# for len in  12 24 48 96
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
#     --e_layers $e_layers \
#     --n_patch 5 \
#     --enc_in 170 \
#     --dec_in 170 \
#     --c_out 170 \
#     --des 'Exp' \
#     --d_model $d_model \
#     --d_ff $d_ff \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --itr 1 \
#     --train_epochs 15 \
#     --use_norm 0
# done

# npatch=5

# long_term_forecast_PEMS08_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl12_dm512_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.0718148946762085, mae:0.16871102154254913, rmse:0.26798301935195923, mape:1.4591275453567505, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS08_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl24_dm512_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.086288683116436, mae:0.18361441791057587, rmse:0.2937493622303009, mape:1.611690878868103, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS08_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl48_dm512_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.11196865886449814, mae:0.20650900900363922, rmse:0.33461716771125793, mape:1.7373013496398926, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001

# long_term_forecast_PEMS08_96_96_Channel_conv_PEMS_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc1_ebtimeF_dtFalse_Exp_0  
# mse:0.15926769375801086, mae:0.2320222705602646, rmse:0.39908355474472046, mape:1.853089690208435, patch:[1, 4, 12, 24], dropout:0.1, d2:0.1, learning_rate:0.001







