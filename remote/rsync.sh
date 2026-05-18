#!/usr/bin/env bash
# rsync.sh push|pull
# push: send code + scripts + configs to the remote (excludes outputs/, .venv/, __pycache__/)
# pull: bring back outputs/ from the remote
set -euo pipefail

REMOTE="${REMOTE:-chen123@d8545-10s10501.wisc.cloudlab.us}"
REMOTE_DIR="${REMOTE_DIR:-~/MoE}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mode="${1:-push}"
case "$mode" in
    push)
        echo "[push] $LOCAL_DIR/ -> $REMOTE:$REMOTE_DIR/"
        rsync -avz --delete \
            --exclude '.venv/' \
            --exclude '.git/' \
            --exclude 'outputs/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude '.idea/' \
            --exclude '.vscode/' \
            "$LOCAL_DIR"/ "$REMOTE:$REMOTE_DIR"/
        ;;
    pull)
        echo "[pull] $REMOTE:$REMOTE_DIR/outputs/ -> $LOCAL_DIR/outputs/"
        mkdir -p "$LOCAL_DIR/outputs"
        rsync -avz "$REMOTE:$REMOTE_DIR/outputs/" "$LOCAL_DIR/outputs/"
        ;;
    *)
        echo "usage: $0 push|pull" >&2
        exit 2
        ;;
esac
