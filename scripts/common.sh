#!/bin/bash
# BabyLM 2026 — Shared shell functions.
# Source this in every script: source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── HF Mirror (auto-detect) ──────────────────────────────────────────
# Test if huggingface.co is reachable; if not, use hf-mirror.com
_setup_hf_mirror() {
    # Already configured? skip
    if [ -n "$HF_ENDPOINT" ]; then
        echo "[HF] Using HF_ENDPOINT=$HF_ENDPOINT"
        return
    fi

    # Check if we can reach HuggingFace
    if curl -s --connect-timeout 5 --max-time 10 https://huggingface.co > /dev/null 2>&1; then
        echo "[HF] huggingface.co reachable, using direct connection"
        return
    fi

    # Use mirror
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "[HF] huggingface.co NOT reachable, using mirror: $HF_ENDPOINT"
}

# ── Proxy (optional) ─────────────────────────────────────────────────
_setup_proxy() {
    # Check common AutoDL proxy ports
    if [ -f "$PROJECT_ROOT/configs/proxy.env" ]; then
        source "$PROJECT_ROOT/configs/proxy.env"
        echo "[Proxy] Loaded from configs/proxy.env: $http_proxy"
    elif [ -n "$AUTODL_PROXY" ]; then
        export http_proxy="$AUTODL_PROXY"
        export https_proxy="$AUTODL_PROXY"
        export HTTP_PROXY="$AUTODL_PROXY"
        export HTTPS_PROXY="$AUTODL_PROXY"
        echo "[Proxy] AUTODL_PROXY=$AUTODL_PROXY"
    else
        # Try auto-detect on common ports
        for port in 7890 10809 8889; do
            if curl -s --connect-timeout 2 "http://127.0.0.1:$port" > /dev/null 2>&1; then
                export http_proxy="http://127.0.0.1:$port"
                export https_proxy="http://127.0.0.1:$port"
                export HTTP_PROXY="http://127.0.0.1:$port"
                export HTTPS_PROXY="http://127.0.0.1:$port"
                echo "[Proxy] Auto-detected at 127.0.0.1:$port"
                return
            fi
        done
        echo "[Proxy] No proxy detected"
    fi
}

# ── Run setup ────────────────────────────────────────────────────────
_setup_hf_mirror
_setup_proxy

# ── Common paths ─────────────────────────────────────────────────────
PYTHON="python -u"
LOG_DIR="$PROJECT_ROOT/logs"
CP_DIR="$PROJECT_ROOT/checkpoints"
CACHE_DIR="$PROJECT_ROOT/data_cache"

mkdir -p "$LOG_DIR"

# ── Logging helpers ──────────────────────────────────────────────────
log()      { echo "[$(date '+%H:%M:%S')] $*"; }
log_ok()   { echo "[$(date '+%H:%M:%S')] ✅ $*"; }
log_fail() { echo "[$(date '+%H:%M:%S')] ❌ $*"; }

# ── Git clone: SSH first (port 22 unfiltered), mirrors as fallback ──
# Usage: git_clone_mirror <url> [target_dir]
git_clone_mirror() {
    local url="$1"
    local target="${2:-}"

    # Quick probe
    if curl -s --connect-timeout 2 --max-time 3 https://github.com > /dev/null 2>&1; then
        echo "[Git] github.com reachable, HTTPS clone..."
        if timeout 10 git clone "$url" "$target" 2>/dev/null; then
            echo "[Git] HTTPS OK"
            return 0
        fi
        [ -n "$target" ] && rm -rf -- "$target" 2>/dev/null || true
    else
        echo "[Git] HTTPS blocked"
    fi

    # SSH won't timeout — port 22 is open
    local ssh_url
    ssh_url=$(echo "$url" | sed 's|https://github.com/|git@github.com:|')
    echo "[Git] Trying SSH: $ssh_url"
    if git clone "$ssh_url" "$target" 2>/dev/null; then
        echo "[Git] SSH OK"
        return 0
    fi
    echo "[Git] SSH failed (no key?), falling back to mirrors..."
    [ -n "$target" ] && rm -rf -- "$target" 2>/dev/null || true

    # Mirrors as last resort
    if timeout 10 git clone "https://ghproxy.com/$url" "$target" 2>/dev/null; then
        echo "[Git] ghproxy OK"
        return 0
    fi

    echo "[Git] ALL FAILED — upload zip via AutoDL file manager"
    return 1
}

# ── Pip install with mirror fallback ──────────────────────────────────
pip_install() {
    # Quick check: can we reach pypi.org?
    if curl -s --connect-timeout 2 --max-time 3 https://pypi.org > /dev/null 2>&1; then
        echo "[Pip] pypi.org reachable, direct install..."
        if pip install "$@"; then
            return 0
        fi
    else
        echo "[Pip] pypi.org NOT reachable, using tsinghua mirror..."
    fi
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "$@"
}
check_gpu() {
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "ERROR: No GPU detected!"
        exit 1
    fi
    python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"
}
