param(
    [string]$EnvPath = '',
    [switch]$Cpu,
    [string]$Extras = '',        # e.g. -Extras 'sam,export,tflite' or 'all'
    [switch]$SkipVerify,
    [switch]$PrintOnly
)
$ErrorActionPreference = 'Stop'
# Keep the environment self-contained: ignore user site-packages so a stale
# ~/.local / %APPDATA% torch cannot shadow (or receive) the project packages.
$env:PYTHONNOUSERSITE = '1'
$env:PIP_USER = '0'
$Project = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) { $EnvPath = Join-Path $Project '.venv-efficientvit' }
$EnvPath = [System.IO.Path]::GetFullPath($EnvPath)
$Python = Join-Path $EnvPath 'python.exe'
$CondaMetadata = Join-Path $EnvPath 'conda-meta'
$TranscriptStarted = $false

function Invoke-Step([string]$Label, [string]$Program, [string[]]$Arguments) {
    Write-Host "`n== $Label =="
    # Activate the target environment for each step (Windows DLL paths).
    if ($Program -eq $Python) {
        $Arguments = @('run', '--no-capture-output', '--prefix', $EnvPath, $Program) + $Arguments
        $Program = 'conda'
    }
    if ($PrintOnly) { Write-Host ($Program + ' ' + ($Arguments -join ' ')); return }
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE). Fix the error and rerun this script." }
}

try {
    if (-not $PrintOnly) {
        if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
            throw 'Conda was not found. Install Miniconda/Anaconda and run from its PowerShell Prompt.'
        }
        if ((Test-Path -LiteralPath $EnvPath) -and -not (Test-Path -LiteralPath $CondaMetadata)) {
            throw "Target exists but is not a Conda environment: $EnvPath. Choose another -EnvPath."
        }
        $LogFolder = Join-Path $Project 'work_dirs/setup'
        New-Item -ItemType Directory -Path $LogFolder -Force | Out-Null
        $LogPath = Join-Path $LogFolder ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
        Start-Transcript -Path $LogPath | Out-Null
        $TranscriptStarted = $true
        Write-Host "Installation log: $LogPath"
    }

    if (-not (Test-Path -LiteralPath $CondaMetadata)) {
        Invoke-Step 'Create Python 3.10 environment' 'conda' @('create', '--prefix', $EnvPath, 'python=3.10', 'pip', '-y')
    }
    Invoke-Step 'Check target Python version' $Python @('-c', 'import sys; assert sys.version_info[:2] == (3, 10)')
    Invoke-Step 'Install package tools' $Python @('-m', 'pip', 'install', 'pip<26', 'setuptools>=65,<81', 'wheel')

    if ($Cpu) {
        $Requirements = Join-Path $Project 'requirements-cpu.txt'
        Invoke-Step 'Install CPU PyTorch + dependencies' $Python @('-m', 'pip', 'install', '-r', $Requirements)
    } else {
        $Requirements = Join-Path $Project 'requirements.txt'
        Invoke-Step 'Install CUDA 11.8 PyTorch + dependencies' $Python @('-m', 'pip', 'install', '-r', $Requirements)
    }

    if ($Extras) {
        $Spec = "$Project[$Extras]"
        Invoke-Step "Register project with extras [$Extras]" $Python @('-m', 'pip', 'install', '-e', $Spec)
    } else {
        Invoke-Step 'Register project (editable)' $Python @('-m', 'pip', 'install', '-e', $Project)
    }
    Invoke-Step 'Verify dependency consistency' $Python @('-m', 'pip', 'check')

    if (-not $SkipVerify) {
        Invoke-Step 'Verify imports and EfficientViT seg model' $Python @((Join-Path $PSScriptRoot 'verify_environment.py'))
    }

    if ($PrintOnly) { Write-Host 'Preview complete; no commands executed and no files written.'; return }
    Write-Host "`nReady. Activate: conda activate '$EnvPath'"
    Write-Host "Seg train : & '$Python' '$(Join-Path $Project 'train_seg_model.py')' --config '$(Join-Path $Project 'configs/seg/crack_b0.json')'"
    Write-Host "Seg eval  : & '$Python' '$(Join-Path $Project 'train_seg_model.py')' --config '$(Join-Path $Project 'configs/seg/crack_b0.json')' --resume <ckpt> --eval_only --test"
    Write-Host "Seg pred  : & '$Python' '$(Join-Path $Project 'predict_seg_model.py')' --config '$(Join-Path $Project 'configs/seg/crack_b0.json')' --checkpoint <ckpt> --input <images> --output predictions"
    Write-Host "Data prep : & '$Python' '$(Join-Path $Project 'prepare_seg_dataset.py')' --root <dataset>"
} finally { if ($TranscriptStarted) { Stop-Transcript | Out-Null } }
