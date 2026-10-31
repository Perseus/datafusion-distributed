#!/usr/bin/env bash
# Interleaved Iceberg dynamic-filter-pruning benchmark. Bench branch only.
# Usage: run.sh <base-bin> <head-bin> <iceberg-dataset> <out-dir> <focused-rounds> <focused-iters> <full-rounds> <full-iters> <row-selection:0|1>
set -uo pipefail
BASE=$1 HEAD=$2 DATA=$3 OUT=$4 FR=$5 FI=$6 SR=$7 SI=$8 RS=$9
mkdir -p "$OUT"

CONFIGS=(base head-off head-on head-on-nostats)
[ "$RS" = 1 ] && CONFIGS+=(head-on-rs)

PERF=0
if perf stat -e task-clock,instructions,cycles -o /dev/null true 2>/dev/null; then PERF=hw
elif perf stat -e task-clock -o /dev/null true 2>/dev/null; then PERF=sw; fi
echo "perf mode: $PERF" | tee "$OUT/perf-mode.txt"

run_cfg() { # cfg phase workers round iters [queries]
  local cfg=$1 phase=$2 w=$3 r=$4 it=$5 q=${6:-}
  local bin=$HEAD
  local -a envs=(env -u DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING -u DFBENCH_ICEBERG_ROW_SELECTION -u DFBENCH_ICEBERG_NO_COLUMN_STATS)
  case $cfg in
    base) bin=$BASE ;;
    head-off) envs+=(DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING=false) ;;
    head-on) envs+=(DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING=true) ;;
    head-on-nostats) envs+=(DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING=true DFBENCH_ICEBERG_NO_COLUMN_STATS=1) ;;
    head-on-rs) envs+=(DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING=true DFBENCH_ICEBERG_ROW_SELECTION=true) ;;
  esac
  local tag="$phase-w$w-$cfg-r$r"
  local -a args=(run --dataset "$DATA" --workers "$w" -i "$it")
  [ -n "$q" ] && args+=(-q "$q")
  local -a wrap=()
  if [ "$phase" = focused ]; then
    case $PERF in
      hw) wrap=(perf stat -x, -e task-clock,instructions,cycles -o "$OUT/$tag.perf") ;;
      sw) wrap=(perf stat -x, -e task-clock -o "$OUT/$tag.perf") ;;
      *) wrap=(/usr/bin/time -v -o "$OUT/$tag.time") ;;
    esac
  fi
  echo "=== $tag $(date -u +%T)"
  "${envs[@]}" "${wrap[@]}" "$bin" "${args[@]}" >"$OUT/$tag.log" 2>&1 || echo "FAILED $tag (exit $?)" | tee -a "$OUT/failures.txt"
}

# Rotate the start config each round so no config always runs first/last.
rotate() { local r=$1; local n=${#CONFIGS[@]}; for ((k = 0; k < n; k++)); do echo "${CONFIGS[$(((k + r) % n))]}"; done; }

for ((r = 1; r <= FR; r++)); do
  for w in 0 4; do
    for cfg in $(rotate "$r"); do run_cfg "$cfg" focused "$w" "$r" "$FI" q8,q9,q17,q20; done
  done
done
for ((r = 1; r <= SR; r++)); do
  for w in 0 4; do
    for cfg in $(rotate "$r"); do run_cfg "$cfg" full "$w" "$r" "$SI"; done
  done
done
