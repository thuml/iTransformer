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


# model_name=Flowformer
$model_name = 'iFlowformer'

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_48_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 48 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_96_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_192_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 192 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_336_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 336 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --itr 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/electricity/ `
  --data_path electricity.csv `
  --model_id ECL_720_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 720 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0005 `
  --itr 1
