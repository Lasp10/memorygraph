#!/bin/bash
set -e

echo "Setting up MemoryGraph..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# ── Dependencies ─────────────────────────────────────────────────────────────
# Face detection/embedding uses OpenCV's DNN module (models/ dir) instead of
# dlib, so there's no native compile step and it installs in seconds via pip.
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ── CLIP model ───────────────────────────────────────────────────────────────
echo "Pre-downloading CLIP model (~300MB, one time only)..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

# ── Directories + DB ─────────────────────────────────────────────────────────
mkdir -p uploads thumbnails
python3 -c "from db.database import init_db; init_db()"

echo ""
echo "Setup complete."
echo ""
echo "To start:"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload"
echo ""
echo "Then open: http://localhost:8000"
