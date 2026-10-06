#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-editing.txt

SAM2_BUILD_CUDA=0 python -m pip install "git+https://github.com/facebookresearch/sam2.git"

mkdir -p third_party
if [ ! -d third_party/ProPainter/.git ]; then
  git clone --depth 1 https://github.com/sczhou/ProPainter.git third_party/ProPainter
else
  git -C third_party/ProPainter pull --ff-only
fi

echo
echo "Editing models are configured."
echo "Grounding DINO and SAM 2 weights download on first use."
echo "ProPainter weights download automatically on first inference."
