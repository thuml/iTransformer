export CUDA_VISIBLE_DEVICES=2

model_name=Reformer

#python -u run.py \
##  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.txt \
#  --model_id solar_96_96 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --learning_rate 0.0005 \
#  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --learning_rate 0.0005 \
  --channel_independence true \
  --exp_name partial_train \
  --batch_size 8 \
  --itr 1

model_name=iReformer

#python -u run.py \
##  --is_training 1 \
#  --root_path ./dataset/Solar/ \
#  --data_path solar_AL.txt \
#  --model_id solar_96_96 \
#  --model $model_name \
#  --data Solar \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 137 \
#  --dec_in 137 \
#  --c_out 137 \
#  --des 'Exp' \
#  --learning_rate 0.0005 \
#  --itr

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 27 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --exp_name partial_train \
  --itr 1