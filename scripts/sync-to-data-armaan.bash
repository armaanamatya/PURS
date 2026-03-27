#!/usr/bin/env bash
set -euo pipefail

# Push PURS experiment tree to the GPU cluster under /data/armaan.
#
# Usage:
#   ./scripts/sync-to-data-armaan.bash
#   ./scripts/sync-to-data-armaan.bash armaan@OTHER_HOST /data/armaan/purs
#
# From Windows:
#   Option A — repo root:     .\scripts\sync-to-data-armaan.ps1
#   Option A — scripts dir:   .\sync-to-data-armaan.ps1
#   Option B (any cwd ok):     .\scripts\sync-to-data-armaan.cmd
#
# Env:
#   DRY_RUN=1          — rsync --dry-run only
#   SYNC_SLIM=1        — skip OmniZip-main/lmms-eval (faster; use if you only run
#                        eval_qwen_omni*.py / viz_*.py, not eval.sh / VideoMME)
#   RSYNC_DELETE=1     — delete extra files on remote (careful)

REMOTE_HOST="${1:-armaan@10.244.120.178}"
REMOTE_DIR="${2:-/data/armaan/purs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RSYNC_OPTS=(
  -avz
  --progress
  --human-readable
  # General junk / tooling
  --exclude=.git/
  --exclude=OmniZip-OG/
  --exclude=website/
  --exclude=node_modules/
  --exclude=__pycache__/
  --exclude='*.py[cod]'
  --exclude=.venv/
  --exclude=venv/
  --exclude=.env
  --exclude=.cursor/
  --exclude=.claude/
  --exclude=.next/
  --exclude='*.egg-info/'
  --exclude='Miniconda3*.sh'
  --exclude=tmp/
  --exclude='*.log'
  --exclude=.DS_Store
  # Non-essential docs / notebooks / PDFs
  --exclude=docs/
  --exclude='*.md'
  --exclude='*.pdf'
  --exclude='*.ipynb'
  # Not needed for PURS experiments
  --exclude=Qwen2.5-Omni/
  --exclude=omnizipresults/
  # Local visualization outputs
  --exclude=attention_viz_qwen/
  --exclude=attention_viz_omnizip/
)

if [[ "${SYNC_SLIM:-0}" == "1" ]]; then
  RSYNC_OPTS+=(--exclude=OmniZip-main/lmms-eval/)
fi

if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_OPTS+=(--delete --delete-excluded)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

echo "Local:  ${REPO_ROOT}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}"
rsync "${RSYNC_OPTS[@]}" "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

cat <<EOF

Done. On the server, typical setup:

  cd ${REMOTE_DIR}/OmniZip-main && bash setup.sh
  cd ${REMOTE_DIR}/OmniZip-main/lmms-eval && pip install -e .

Set model path (or download Qwen2.5-Omni-7B) and point scripts at:
  MODEL:   e.g. ${REMOTE_DIR}/model  or  HuggingFace cache
  videos:  ${REMOTE_DIR}/videos
  metadata: ${REMOTE_DIR}/videos/metadata.json

OmniZip eval example:
  cd ${REMOTE_DIR}
  python eval_qwen_omni_zip.py --metadata videos/metadata.json --videos videos --output results_zip.jsonl
EOF
