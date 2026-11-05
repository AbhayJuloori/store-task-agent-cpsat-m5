#!/usr/bin/env bash
# Packs code, scenarios and agent.duckdb into a private Kaggle dataset, then pushes the GPU kernel.
set -euo pipefail
cd "$(dirname "$0")/../.."
USER_NAME=$(kaggle config view 2>/dev/null | awk -F': ' '/username/ {print $2}')
DS="$USER_NAME/store-task-agent-bundle"
STAGE=$(mktemp -d)
mkdir -p "$STAGE/dataset" "$STAGE/kernel" "$STAGE/bundle/data/work" "$STAGE/bundle/eval"
cp -R src configs "$STAGE/bundle/"
cp eval/scenarios.jsonl "$STAGE/bundle/eval/"
cp data/work/agent.duckdb "$STAGE/bundle/data/work/"
find "$STAGE/bundle" -name __pycache__ -prune -exec rm -rf {} +
tar -czf "$STAGE/dataset/bundle.tar.gz" -C "$STAGE/bundle" .
cat > "$STAGE/dataset/dataset-metadata.json" <<JSON
{"title": "store-task-agent-bundle", "id": "$DS", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status "$DS" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/dataset" -m "update bundle" -q
else
  kaggle datasets create -p "$STAGE/dataset" -q
fi
cp scripts/kaggle/agent_eval_kernel.py "$STAGE/kernel/agent_eval_kernel.py"
cat > "$STAGE/kernel/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/store-task-agent-eval", "title": "store-task-agent-eval", "code_file": "agent_eval_kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["$DS"]}
JSON
echo "waiting for the dataset to be ready"
sleep 60
until kaggle datasets status "$DS" 2>/dev/null | grep -q ready; do sleep 10; done
kaggle kernels push -p "$STAGE/kernel"
