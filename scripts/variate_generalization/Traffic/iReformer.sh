export CUDA_VISIBLE_DEVICES=2

model_name=Reformer

#python -u run.py \
##  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 172 \
  --dec_in 172 \
  --c_out 172 \
  --des 'Exp' \
  --channel_independence true \
  --exp_name partial_train \
  --batch_size 4 \
  --d_model 32 \
  --d_ff 64 \
  --itr 1

model_name=iReformer

#python -u run.py \
##  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 172 \
  --dec_in 172 \
  --c_out 172 \
  --des 'Exp' \
  --exp_name partial_train \
  --itr 1