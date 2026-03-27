#!/usr/bin/env pwsh
<# PowerShell version of mfITransformer.sh
    Places the working dir at project root and runs the same python command.
    Uses a concrete Python interpreter path when possible to avoid env mismatch.
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir '..\..\..')).Path
Set-Location $ProjectRoot

# Prefer active conda env python, otherwise try sibling iTransformer env, then fallback to PATH python.
$python = $null
$assignmentRoot = Split-Path $ProjectRoot -Parent
$fallbackCondaPython = Join-Path (Join-Path (Join-Path $assignmentRoot 'iTransformer') '.conda') 'python.exe'
if (Test-Path $fallbackCondaPython) {
    $python = $fallbackCondaPython
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX 'python.exe'))) {
    $python = Join-Path $env:CONDA_PREFIX 'python.exe'
} else {
    $python = 'python'
}
$args = @(
    '-u','run.py',
    '--is_training','1',
    '--root_path','./dataset/Beijing_Air',
    '--data_path','PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    '--model_id','Beijing_MF_Aotizhongxin',
    '--model','MfITransformer',
    '--data','Beijing_MF',
    '--features','M',
    '--target','PM2.5',
    '--freq','h',
    '--seq_len','96',
    '--label_len','48',
    '--pred_len','24',
    '--enc_in','11',
    '--dec_in','11',
    '--c_out','11',
    '--d_model','256',
    '--n_heads','8',
    '--e_layers','2',
    '--d_layers','1',
    '--d_ff','512',
    '--dropout','0.1',
    '--des','Exp',
    '--itr','1',
    '--exp_name', 'multi_train',
    '--batch_size','16',
    '--learning_rate','0.0001',
    '--train_epochs','10',
    '--patience','3',
    '--mf_freqs','1h,1d',
    '--mf_seq_lens','96,7',
    '--mf_pred_lens','24,2',
    '--mf_var_groups','1h:TEMP|PRES|DEWP|RAIN|WSPM;1d:PM2.5|PM10|SO2|NO2|CO|O3',
    '--mf_target_groups','1h:TEMP|PRES|DEWP|RAIN|WSPM;1d:PM2.5|PM10|SO2|NO2|CO|O3',
    '--mf_loss_weights','1h:1.0;1d:1.0'
)

Write-Host "Running experiment from" $ProjectRoot
Write-Host "Using python:" $python
& $python @args
