# Example: run llama.cpp llama-server for fixed-length entropy collection (ignore_eos via API body).
# 1) Build or download llama-server from https://github.com/ggml-org/llama.cpp/releases
# 2) Obtain a GGUF weights file (same family as your Ollama model if you want comparability).
# 3) Start the server with context >= prompt_tokens + fixed_tokens + margin (e.g. -c 32768).
#
# Then from repo root:
#   python code/collect_llm_entropy_series.py --backend llama-server --quick --fixed-tokens 1024
#
# Default install folder (edit if you move binaries / GGUF).
$LlamaDir = "C:\Users\csapl\OneDrive\Asztali gép\Dani dok\llama"

$ErrorActionPreference = "Stop"
$LlamaServer = Join-Path $LlamaDir "llama-server.exe"
$Gguf = Join-Path $LlamaDir "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
$Ctx = 32768
$Port = 8080

if (-not (Test-Path $LlamaServer)) { throw "Edit LlamaServer path in run_llama_server_example.ps1" }
if (-not (Test-Path $Gguf)) { throw "Edit Gguf path in run_llama_server_example.ps1" }

Write-Host "Starting llama-server on 127.0.0.1:$Port (ctx=$Ctx)..."
Write-Host "Stop with Ctrl+C, then run collect_llm_entropy_series.py with --backend llama-server"
Push-Location $LlamaDir
try {
    & $LlamaServer -m $Gguf --port $Port --host 127.0.0.1 -c $Ctx
} finally {
    Pop-Location
}
