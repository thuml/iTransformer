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


$model_name = 'iTransformer'

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/PEMS/ `
  --data_path PEMS08.npz `
  --model_id PEMS08_96_12 `
  --model $model_name `
  --data PEMS `
  --features M `
  --seq_len 96 `
  --pred_len 12 `
  --e_layers 2 `
  --enc_in 170 `
  --dec_in 170 `
  --c_out 170 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --itr 1 `
  --use_norm 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/PEMS/ `
  --data_path PEMS08.npz `
  --model_id PEMS08_96_24 `
  --model $model_name `
  --data PEMS `
  --features M `
  --seq_len 96 `
  --pred_len 24 `
  --e_layers 2 `
  --enc_in 170 `
  --dec_in 170 `
  --c_out 170 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --itr 1 `
  --use_norm 1

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/PEMS/ `
  --data_path PEMS08.npz `
  --model_id PEMS08_96_48 `
  --model $model_name `
  --data PEMS `
  --features M `
  --seq_len 96 `
  --pred_len 48 `
  --e_layers 4 `
  --enc_in 170 `
  --dec_in 170 `
  --c_out 170 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --itr 1 `
  --use_norm 0

python -u run.py `
  --is_training 1 `
  --root_path ./dataset/PEMS/ `
  --data_path PEMS08.npz `
  --model_id PEMS08_96_96 `
  --model $model_name `
  --data PEMS `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 4 `
  --enc_in 170 `
  --dec_in 170 `
  --c_out 170 `
  --des 'Exp' `
  --d_model 512 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.001 `
  --itr 1 `
  --use_norm 0
