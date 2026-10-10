#!/usr/bin/env bash
# Run one closed-loop job: run_one.sh <tag> <trace.json> <policy> [extra run_policy args...]
# Result directory is printed in logs/batch/<tag>/<trace>__<policy>.log ("results in ...").
# A job whose log already contains "results in" is skipped (resumable batches).
set -u
TAG=$1; TRACE=$2; POLICY=$3; shift 3
ROOT=/c/Users/kanif/work/Pandia
NAME=$(basename "$POLICY" .onnx)
LOG=$ROOT/localtest/logs/batch/$TAG/${TRACE%.json}__${NAME}.log
mkdir -p "$(dirname "$LOG")"
if [ -f "$LOG" ] && grep -q "results in" "$LOG"; then exit 0; fi
CN=pdrv_${TAG}_${TRACE%.json}_${NAME//[^A-Za-z0-9]/_}
MSYS_NO_PATHCONV=1 docker run --rm --name "$CN" \
  -v /run/desktop/mnt/host/c/Users/kanif/work/Pandia:/data2/kj/Workspace/Pandia \
  -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp \
  -e HOST_PANDIA=/run/desktop/mnt/host/c/Users/kanif/work/Pandia \
  -e PYTHONPATH=/data2/kj/Workspace/Pandia -e PYTHONUNBUFFERED=1 \
  -w /data2/kj/Workspace/Pandia pandia-driver:local \
  python localtest/run_policy.py "$TRACE" --policy "$POLICY" --tag "$TAG" "$@" > "$LOG" 2>&1
echo "$(date +%T) done $TAG $TRACE $NAME $(grep -c 'results in' "$LOG")"
