#!/usr/bin/env pwsh
<#
PowerShell setup helper for experiment scripts.
- Resolves project root
- Prefers an active Conda env
- Falls back to sibling iTransformer/.conda python if available
- Sets CUDA visibility default
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir '..')

Write-Host "Checking environment setup..."

$assignmentRoot = Split-Path $ProjectRoot -Parent
$fallbackCondaPython = Join-Path (Join-Path (Join-Path $assignmentRoot 'iTransformer') '.conda') 'python.exe'

if (-not $env:CONDA_PREFIX -and (Test-Path $fallbackCondaPython)) {
    Write-Host "Conda environment not active in current shell."
    Write-Host "Using fallback Python:" $fallbackCondaPython
    $env:PYTHON_EXECUTABLE = $fallbackCondaPython
} elseif ($env:CONDA_PREFIX) {
    $activePython = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path $activePython) {
        Write-Host "Using active Conda environment:" $env:CONDA_PREFIX
        $env:PYTHON_EXECUTABLE = $activePython
    }
}

if (-not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = '0'
}

Write-Host "Environment is ready."
Write-Host "=========================================="
