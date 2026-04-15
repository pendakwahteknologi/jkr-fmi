#!/bin/bash
# =============================================================================
# JKR AI — GX10 Setup Script
# =============================================================================

set -euo pipefail

PROJECT_DIR="/opt/jkr-ai"
FRONTEND_DIR="/var/www/jkr-ai/public"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/venv"
USER_NAME="$(whoami)"

echo "============================================"
echo "  JKR AI — GX10 Setup"
echo "============================================"

# ---------- Platform detection ----------
echo ""
echo "[1/7] Detecting platform..."

ARCH=$(uname -m)
GPU_NAME="none"
CUDA_VERSION="none"
GPU=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' 2>/dev/null || echo "unknown")
    echo "  GPU: $GPU_NAME"
    echo "  CUDA: $CUDA_VERSION"
    GPU=true
else
    echo "  No GPU detected — CPU-only mode"
fi

CPU_CORES=$(nproc)
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "  CPU cores: $CPU_CORES"
echo "  RAM: ${TOTAL_RAM_GB}GB"
echo "  Architecture: $ARCH"

WORKERS=$(( CPU_CORES > 4 ? 4 : CPU_CORES ))

# ---------- Create directories ----------
echo ""
echo "[2/7] Creating directories..."
sudo mkdir -p "$PROJECT_DIR"/{logs,knowledge,documents,chroma_db}
sudo mkdir -p "$FRONTEND_DIR"
sudo chown -R "$USER_NAME":"$USER_NAME" "$PROJECT_DIR"

# ---------- Copy files ----------
echo ""
echo "[3/7] Copying files..."
cp "$REPO_DIR"/backend/app.py "$PROJECT_DIR/"
cp "$REPO_DIR"/backend/providers.py "$PROJECT_DIR/"
cp "$REPO_DIR"/backend/agency_config.py "$PROJECT_DIR/"

sudo cp "$REPO_DIR"/frontend/*.html "$FRONTEND_DIR/"
sudo cp "$REPO_DIR"/frontend/*.png "$FRONTEND_DIR/" 2>/dev/null || true
sudo cp "$REPO_DIR"/frontend/*.jpg "$FRONTEND_DIR/" 2>/dev/null || true

cp "$REPO_DIR"/knowledge/* "$PROJECT_DIR/knowledge/" 2>/dev/null || true

# ---------- Python virtual environment ----------
echo ""
echo "[4/7] Creating Python virtual environment..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools --quiet

# ---------- Install PyTorch ----------
echo ""
echo "[5/7] Installing PyTorch..."
if [ "$GPU" = true ]; then
    echo "  Installing GPU-accelerated PyTorch..."
    pip install --no-cache-dir torch torchvision 2>&1 | tail -5
    python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
" || echo "  Warning: CUDA verification failed"
else
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
fi

# ---------- Install dependencies ----------
echo ""
echo "[6/7] Installing Python dependencies..."
pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pydantic python-dotenv \
    openai \
    chromadb sentence-transformers rank-bm25 \
    pymupdf python-docx \
    orjson \
    2>&1 | tail -10

echo "  Dependencies installed"

# ---------- Pre-download ML models ----------
echo ""
echo "[7/7] Pre-downloading ML models..."

export HF_HOME="$PROJECT_DIR/.hf_cache"
mkdir -p "$HF_HOME"

python3 -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'

print('  Downloading Mesolitica embedding model...')
from sentence_transformers import SentenceTransformer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('mesolitica/mistral-embedding-191m-8k-contrastive', device=device)
_ = model.encode(['test'], device=device)
print(f'  Embedding model ready on {device}')

print('  Downloading cross-encoder model...')
from sentence_transformers import CrossEncoder
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
_ = ce.predict([('test query', 'test document')])
print('  Cross-encoder model ready')

print('  All models pre-downloaded and verified')
"

# ---------- Environment file ----------
if [ ! -f "$PROJECT_DIR/backend.env" ]; then
    cp "$REPO_DIR/configs/backend.env.template" "$PROJECT_DIR/backend.env"
    echo "  Created backend.env — edit it to add MTAI API keys"
fi

# ---------- systemd ----------
echo "  Installing systemd service..."
sudo cp "$REPO_DIR/configs/jkr-ai.service" /etc/systemd/system/
sudo systemctl daemon-reload
echo "  systemd service installed"

# ---------- nginx ----------
if command -v nginx &>/dev/null; then
    echo "  Installing nginx config..."
    sudo cp "$REPO_DIR/configs/jkr-ai.conf" /etc/nginx/sites-enabled/
    sudo nginx -t 2>&1 && sudo systemctl reload nginx
    echo "  nginx configured"
fi

# ---------- Summary ----------
echo ""
echo "============================================"
echo "  JKR AI Setup Complete!"
echo "============================================"
echo ""
echo "  Hardware: $GPU_NAME | $CPU_CORES cores | ${TOTAL_RAM_GB}GB RAM"
echo "  Workers: $WORKERS uvicorn processes"
echo "  Port: 8003"
echo ""
echo "  Next steps:"
echo "    1. Edit $PROJECT_DIR/backend.env with MTAI API keys"
echo "    2. Run ingestion: bash $REPO_DIR/scripts/ingest.sh"
echo "    3. Start: sudo systemctl start jkr-ai"
echo "    4. Check: curl http://localhost:8003/api/health"
echo ""
