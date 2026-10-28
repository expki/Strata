#!/bin/bash
# The box's A/B kit, sourced by a run script that itself runs under flock ~/bench-mixed/.box.lock.
# The caller sets STEP (the file prefix, for example t21-s1) and may set PREP, PASSES, STARTS and PL before sourcing.
#   power                       set the 3090 to $PL W (350) with its clocks pinned at $LGC and $LMC MHz, the W6800 at its top clock, and check them
#   at TREE SHA                 the tree is at SHA with no local changes in include, src, tests or CMakeLists.txt
#   prep keep|rebuild           keep the page cache, or drop it and read the five shards back in order
#   start ARM                   one engine start with box_bench.py, ARM is NAME|CONFIG|KV|EXPECT
#   abba ARM_A ARM_B [N]        N starts of each arm: A B for 1, A B B A for 2, A B A B B A for 3
#   verdict A B TARGET [OTHER]  box_bench.py verdict over $STEP.jsonl
# CONFIG is a file in ~/bench-mixed or a path, its cwd is the tree whose server runs the arm.
# KV is box_bench.py's arm keys (env, xargs), EXPECT its --expect facts.
# Each start sets the power and the locks, runs prep and fails when the sampler saw the 3090 above its pinned clock.
B=$HOME/bench-mixed
LOCK=$B/.box.lock
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-$HOME/source/strata/.venv/bin/python}
PREP=${PREP:-keep}
# the user pins the 3090 at 1500 MHz and keeps the W6800 at its top clock with no idling (2026-10-27)
LGC=${LGC:-1500,1500}
LMC=${LMC:-9751,9751}
W6800_DEV=${W6800_DEV:-/sys/bus/pci/devices/0000:c8:00.0}
W6800_GFXOFF=${W6800_GFXOFF:-/sys/kernel/debug/dri/1/amdgpu_gfxoff}
export BOX_CLK_MAX=${BOX_CLK_MAX:-${LGC#*,}}
STARTS=${STARTS:-2}
PASSES=${PASSES:-"--pass warm:chat0,p20k --pass m:chat0,chat1,chat2,p20k --pass m:chat0,chat1,chat2,p20k"}
K=0
[ -n "$STEP" ] || { echo "== set STEP before sourcing box_ab.sh"; exit 2; }

under_lock() {   # the lock is held and a parent of this shell holds it
  flock -n "$LOCK" true && return 1
  local p=$$
  while [ "${p:-1}" -gt 1 ]; do
    [ "$(cat /proc/$p/comm)" = flock ] && grep -qF .box.lock /proc/$p/cmdline && return 0
    p=$(ps -o ppid= -p "$p" | tr -d ' ')
  done
  return 1
}
under_lock || { echo "== not run under flock $LOCK"; exit 5; }
[ -e "$B/$STEP.jsonl" ] && { echo "== $B/$STEP.jsonl exists: pick another STEP"; exit 6; }
export TMPDIR=$B/$STEP-tmp
mkdir -p "$TMPDIR"
systemctl is-active --quiet strata.service && { echo "== strata.service is running, stopping it $(date)"; sudo systemctl stop strata.service; }
echo $$ > "$B/$STEP-run.pid"
sudo prlimit --pid $$ --memlock=unlimited:unlimited
trap "exit 1" TERM INT HUP

power() {
  sudo nvidia-smi -i 0 -pl "${PL:-350}" > /dev/null && sudo nvidia-smi -i 0 -lgc "$LGC" > /dev/null &&
    sudo nvidia-smi -i 0 -lmc "$LMC" > /dev/null || return 1
  w6800_pin || return 1
  local c
  c=$(nvidia-smi --query-gpu=index,power.limit,clocks.sm --format=csv,noheader | tr '\n' ' ')
  echo "power: $c, clocks pinned at $LGC and $LMC, the W6800 at its top clock"
  case "$c" in "0, ${PL:-350}.00 W, ${LGC%,*} MHz "*) ;; *) echo "== the 3090 is not at ${PL:-350} W and ${LGC%,*} MHz"; return 1 ;; esac
}
# the W6800 in manual mode at its top shader and memory levels, with GFXOFF off so it never sleeps between calls
w6800_pin() {
  local top
  echo manual | sudo tee "$W6800_DEV/power_dpm_force_performance_level" > /dev/null || return 1
  top=$(( $(grep -c . "$W6800_DEV/pp_dpm_sclk") - 1 ))
  echo "$top" | sudo tee "$W6800_DEV/pp_dpm_sclk" > /dev/null
  top=$(( $(grep -c . "$W6800_DEV/pp_dpm_mclk") - 1 ))
  echo "$top" | sudo tee "$W6800_DEV/pp_dpm_mclk" > /dev/null
  printf '\x00\x00\x00\x00' | sudo tee "$W6800_GFXOFF" > /dev/null
  [ "$(cat "$W6800_DEV/power_dpm_force_performance_level")" = manual ] || { echo "== the W6800 is not in manual"; return 1; }
  grep -q '500Mhz \*' "$W6800_DEV/pp_dpm_sclk" && { echo "== the W6800 idles at 500 MHz"; return 1; }
  return 0
}
at() {
  [ "$(git -C "$1" rev-parse --short=8 HEAD)" = "$2" ] || { echo "== $1 is not at $2 $(date)"; return 1; }
  [ -z "$(git -C "$1" status --short -- include src tests CMakeLists.txt)" ] || { echo "== $1 has local changes $(date)"; return 1; }
}
warm() {
  local d0 d1 t0
  d0=$(awk '$3 ~ /^nvme[0-9]n1$/ {s += $6} END {print s}' /proc/diskstats); t0=$(date +%s)
  cat /opt/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-Q6_K-0000?-of-00005.gguf > /dev/null
  d1=$(awk '$3 ~ /^nvme[0-9]n1$/ {s += $6} END {print s}' /proc/diskstats)
  echo "warm: shards read through in $(( $(date +%s) - t0 )) s, $(( (d1 - d0) * 512 / 1000000 )) MB from the drives, $(grep ^Cached /proc/meminfo)"
}
prep() {
  case "$1" in
    keep) ;;
    rebuild) sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null && warm ;;
    *) echo "== prep keep|rebuild, not $1"; return 1 ;;
  esac
}
start() {
  local name cfg kv exp root rc
  IFS='|' read -r name cfg kv exp <<< "$1"
  case "$cfg" in /*) ;; *) cfg=$B/$cfg ;; esac
  power && prep "$PREP" || return 1
  root=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["cwd"])' "$cfg") || return 1
  K=$((K + 1))
  echo "######## $STEP: start $K $name $(basename "$cfg") $(date) $(date +%s)"
  echo "meminfo before: $(grep -E '^(MemFree|MemAvailable|Cached)' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  # shellcheck disable=SC2086
  nice -n 5 $PY -u "$KIT/box_bench.py" "$root" "$cfg" "$name:$kv" --log "$B/$STEP.jsonl" --keep-logs "$B/$STEP-$K-" \
    --sample ${exp:+--expect "$exp"} $PASSES 2>&1 | tee "$TMPDIR/start-$K.out" |
    while IFS= read -r l; do echo "$(date +%s.%N | cut -c1-14) $l"; done
  rc=${PIPESTATUS[0]}
  echo "bench exit $rc $(date +%s)"
  grep -q CLOCK_OVER "$TMPDIR/start-$K.out" && { echo "== the 3090 ran above $BOX_CLK_MAX MHz in start $K"; rc=7; }
  return "$rc"
}
abba() {
  local order="" i
  [ $(( ${3:-$STARTS} % 2 )) -eq 1 ] && order=AB
  for ((i = 0; i < ${3:-$STARTS} / 2; i++)); do order=${order}ABBA; done
  echo "== $STEP: order $order"
  for ((i = 0; i < ${#order}; i++)); do
    if [ "${order:i:1}" = A ]; then start "$1" || return; else start "$2" || return; fi
  done
}
verdict() {
  local a=$1 b=$2 t=$3 o=() m
  shift 3
  for m in "$@"; do o+=(--other "$m"); done
  $PY "$KIT/box_bench.py" verdict "$B/$STEP.jsonl" --a "$a" --b "$b" --target "$t" "${o[@]}"
}

power || exit 1
nvidia-smi dmon -s puct -d 1 -o T > "$B/$STEP-dmon.txt" 2>&1 &
DM=$!
trap "kill $DM 2>/dev/null; echo \"== run ends \$(date) \$(date +%s)\"" EXIT
