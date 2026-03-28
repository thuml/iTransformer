#!/usr/bin/env pwsh
<# Auto-generated PowerShell equivalent of the original bash script. #>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = $ScriptDir
for ($i = 0; $i -lt 8; $i++) {
    if (Test-Path (Join-Path $ProjectRoot 'run.py')) { break }
    $ProjectRoot = Split-Path $ProjectRoot -Parent
}
if (-not (Test-Path (Join-Path $ProjectRoot 'run.py'))) { throw 'Could not locate project root (run.py).' }
Set-Location $ProjectRoot
. (Join-Path $ProjectRoot 'scripts/setup_env.ps1')


$model_name = 'Flowformer'

#python -u run.py `
##  --is_training 1 `
#  --root_path ./dataset/Solar/ `
#  --data_path solar_AL.txt `
#  --model_id solar_96_96 `
#  --model $model_name `
#  --data Solar `
#  --features M `
#  --seq_len 96 `
#  --label_len 48 `
#  --pred_len 96 `
#  --e_layers 2 `
#  --d_layers 1 `
#  --factor 3 `
#  --enc_in 137 `
#  --dec_in 137 `
#  --c_out 137 `
#  --des 'Exp' `
#  --learning_rate 0.0005 `
#  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/Solar/ `
  --data_path solar_AL.txt `
  --model_id solar_96_96 `
  --model $model_name `
  --data Solar `
  --features M `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_layers 1 `
  --factor 3 `
  --enc_in 27 `
  --dec_in 27 `
  --c_out 27 `
  --des 'Exp' `
  --d_model 32 `
  --d_ff 64 `
  --learning_rate 0.0005 `
  --channel_independence true `
  --exp_name partial_train `
  --batch_size 8 `
  --itr 1

$model_name = 'iFlowformer'

#python -u run.py `
##  --is_training 1 `
#  --root_path ./dataset/Solar/ `
#  --data_path solar_AL.txt `
#  --model_id solar_96_96 `
#  --model $model_name `
#  --data Solar `
#  --features M `
#  --seq_len 96 `
#  --label_len 48 `
#  --pred_len 96 `
#  --e_layers 2 `
#  --d_layers 1 `
#  --factor 3 `
#  --enc_in 137 `
#  --dec_in 137 `
#  --c_out 137 `
#  --des 'Exp' `
#  --learning_rate 0.0005 `
#  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/Solar/ `
  --data_path solar_AL.txt `
  --model_id solar_96_96 `
  --model $model_name `
  --data Solar `
  --features M `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_layers 1 `
  --factor 3 `
  --enc_in 27 `
  --dec_in 27 `
  --c_out 27 `
  --des 'Exp' `
  --learning_rate 0.0005 `
  --exp_name partial_train `
  --itr 1
