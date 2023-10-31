export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1