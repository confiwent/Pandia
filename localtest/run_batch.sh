#!/usr/bin/env bash
# run_batch.sh <tag> <jobs file: "trace policy [args]" per line> <parallel>
TAG=$1; JOBS=$2; P=${3:-4}
cd /c/Users/kanif/work/Pandia/localtest
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | xargs -P "$P" -L 1 ./run_one.sh "$TAG"
