$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $workspace '.venv-voice'
$python = Join-Path $venv 'Scripts\python.exe'
$modelsRoot = Join-Path $PSScriptRoot 'models'
$piperRoot = Join-Path $modelsRoot 'piper'
$sttRoot = Join-Path $modelsRoot 'stt'

New-Item -ItemType Directory -Force -Path $piperRoot | Out-Null
New-Item -ItemType Directory -Force -Path $sttRoot | Out-Null

if (-not (Test-Path $python)) {
  py -3.14 -m venv $venv
}

& $python -m pip install --upgrade pip
& $python -m pip install faster-whisper piper-tts sounddevice websocket-client
& $python -m piper.download_voices --data-dir $piperRoot en_US-lessac-medium

$warmup = @'
from faster_whisper import WhisperModel
WhisperModel("small.en", device="cpu", compute_type="int8", download_root=r"__STT_ROOT__")
print("Whisper model ready")
'@.Replace('__STT_ROOT__', $sttRoot.Replace('\', '\\'))

& $python -c $warmup
Write-Host 'Voice setup complete.'
