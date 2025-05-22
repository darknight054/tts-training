#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
MODEL_NAME=$(basename "$(dirname "$CONFIG")")
PORT=10902

while :; do
  torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py --c "$CONFIG" --model "$MODEL_NAME"
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training finished successfully—exiting loop."
    break
  else
    echo "⚠️  Crashed with exit code $EXIT_CODE. Cleaning up and retrying in 30s…"
    # kill any stray processes
    for PID in $(pgrep -f "python.*$CONFIG"); do
      echo "Killing stray PID $PID"
      kill -9 $PID
    done
    sleep 30
  fi
done
