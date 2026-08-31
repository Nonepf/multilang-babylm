#!/bin/bash
set -e

if [ "$1" != "--confirm-shutdown" ]; then
    echo "Refusing to arm automatic shutdown without explicit confirmation."
    echo "Usage: bash scripts/auto_shutdown.sh --confirm-shutdown"
    exit 2
fi

echo "[$(date '+%H:%M:%S')] Watching for training processes..."
echo "  Will auto-shutdown when all 'src.train' processes finish."
echo "  Press Ctrl-C to cancel."
echo ""

while true; do
    PIDS=$(pgrep -f "src.train\|python.*train\.py" 2>/dev/null)

    if [ -z "$PIDS" ]; then
        echo "[$(date '+%H:%M:%S')] No training processes found — shutting down!"
        sleep 5
        sudo shutdown -h now
        exit 0
    fi

    echo "[$(date '+%H:%M:%S')] Training still running (PIDs: $(echo $PIDS | tr '\n' ' '))"
    sleep 30
done
