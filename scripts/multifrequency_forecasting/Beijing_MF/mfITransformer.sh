#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." >/dev/null 2>&1 && pwd )"
source "$PROJECT_ROOT/scripts/setup_env.sh"

cd "$PROJECT_ROOT"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Beijing_Air \
  --data_path PRSA_Data_Aotizhongxin_20130301-20170228.csv \
  --model_id Beijing_MF_Aotizhongxin \
  --model MfITransformer \
  --data Beijing_MF \
  --features M \
  --target PM2.5 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --dropout 0.1 \
  --des Exp \
  --itr 1 \
  --exp_name multi_train \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --mf_freqs 1h,1d \
  --mf_seq_lens 96,4 \
  --mf_pred_lens 24,1 \
  --mf_var_groups '1h:TEMP|PRES|DEWP|RAIN|WSPM;1d:PM2.5|PM10|SO2|NO2|CO|O3' \
  --mf_target_groups '1h:TEMP|PRES|DEWP|RAIN|WSPM;1d:PM2.5|PM10|SO2|NO2|CO|O3' \
  --mf_loss_weights '1h:1.0;1d:1.0'
