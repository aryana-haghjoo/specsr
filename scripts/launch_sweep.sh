#!/usr/bin/env bash
#
# launch_sweep.sh — create a W&B sweep and start its agent, in one command.
#
#   ./scripts/launch_sweep.sh sr1                  # preflight, create, launch
#   ./scripts/launch_sweep.sh sr1 --preflight      # checks only, launches nothing
#   ./scripts/launch_sweep.sh sr1 --sweep-id ab12cd34   # attach to an existing sweep
#   ./scripts/launch_sweep.sh zhead
#   ./scripts/launch_sweep.sh sr2
#
# Design notes
# ------------
# * The agent runs inside `screen`, so the sweep survives logoff, and is wrapped
#   in a notifier when one is available (see `notify_wrapper` below) so it
#   emails start and finish with the W&B link. `python -u` either way, or the
#   link may not be scraped before the start notification is sent.
# * **Check whether the GPU is free before launching.** Preflight refuses to
#   start when another process holds it: a multi-day sweep landing on top of
#   someone else's job slows both and can exhaust device memory. Override only
#   when the other process is known to be yours (SPECSR_IGNORE_BUSY_GPU=1).
# * Preflight runs the sweep-config test suite before anything is created. Those
#   tests are what stand between you and a sweep that searches names the trainer
#   never reads: eight dead axes out of eleven is a real configuration this repo
#   has held, and it produces a finished sweep whose parameter-importance plot
#   is noise with a plausible shape.
# * Nothing here writes to `runs/<stage>`: each trial gets its own directory
#   from `WANDB_RUN_ID` (see specsr.training.common.resolve_out_dir), so trials
#   cannot overwrite each other.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Prefer the checkout's own virtualenv when there is one, so the script works
# whether or not the environment happens to be activated.
if [[ -z "${SPECSR_PYTHON:-}" && -x "$REPO/sup_res/bin/python" ]]; then
  PY="$REPO/sup_res/bin/python"
else
  PY="${SPECSR_PYTHON:-$(command -v python3 || command -v python)}"
fi
DATASET="${SPECSR_DATASET:-$REPO/data/paired_DR4_logR.npz}"
PROJECT="${SPECSR_WANDB_PROJECT:-spectral-superresolution}"
MIN_DISK_GB="${SPECSR_MIN_DISK_GB:-50}"

STAGE="${1:-}"
shift || true
PREFLIGHT_ONLY=0
SWEEP_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight) PREFLIGHT_ONLY=1 ;;
    --sweep-id)  SWEEP_ID="${2:?--sweep-id needs a value}"; shift ;;
    --sweep-id=*) SWEEP_ID="${1#*=}" ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$STAGE" in
  sr1|zhead|sr2) ;;
  *) echo "usage: $0 <sr1|zhead|sr2> [--preflight] [--sweep-id ID]" >&2; exit 2 ;;
esac

SWEEP_CFG="$REPO/configs/sweeps/${STAGE}.yaml"
TAG="sweep_${STAGE}"

# How runs get their start/finish notifications, in priority order:
#   1. $SPECSR_NOTIFY_CMD, if you point it at your own wrapper
#   2. `train-notify` on PATH, for an existing local setup
#   3. scripts/notify-run, which ships with the repo and is opt-in: unconfigured
#      it is a transparent passthrough, so wrapping costs nothing
# Configure notify-run by writing ~/.specsr_notify.conf with your own address
# and SMTP details; see the header of scripts/notify-run, or run it --check.
notify_wrapper() {
  if [[ -n "${SPECSR_NOTIFY_CMD:-}" ]]; then
    printf '%s' "$SPECSR_NOTIFY_CMD"
  elif command -v train-notify >/dev/null; then
    printf '%s' train-notify
  elif [[ -x "$REPO/scripts/notify-run" ]]; then
    printf '%s' "$REPO/scripts/notify-run"
  fi
}

log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }

preflight() {
  log "Preflight: $STAGE"
  local errs=0

  [[ -x "$PY" ]] && ok "python: $PY" \
    || { echo "  MISSING python: $PY"; errs=$((errs+1)); }

  [[ -f "$SWEEP_CFG" ]] && ok "sweep config: configs/sweeps/${STAGE}.yaml" \
    || { echo "  MISSING sweep config: $SWEEP_CFG"; errs=$((errs+1)); }

  [[ -f "$DATASET" ]] && ok "dataset: $(basename "$DATASET") ($(du -h "$DATASET" | cut -f1))" \
    || { echo "  MISSING dataset: $DATASET"; errs=$((errs+1)); }

  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ok "CUDA: $("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))')"
  else
    echo "  NO CUDA -- a sweep would run on CPU and never finish"; errs=$((errs+1))
  fi

  # Shared machine: do not start a multi-day job on someone else's GPU.
  # `|| true` is essential -- under `set -e` a failing command substitution kills
  # the script silently, with no error message at all.
  local busy; busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  if [[ -z "$busy" ]]; then
    ok "GPU idle -- safe to launch"
  elif [[ -n "${SPECSR_IGNORE_BUSY_GPU:-}" ]]; then
    ok "GPU busy (pids: $(echo "$busy" | tr '\n' ' ')) -- overridden by SPECSR_IGNORE_BUSY_GPU"
  else
    echo "  GPU IS IN USE by pid(s): $(echo "$busy" | tr '\n' ' ')"
    echo "  This machine is shared. Confirm the job is yours, then re-run with"
    echo "  SPECSR_IGNORE_BUSY_GPU=1, or wait for the GPU to free."
    errs=$((errs+1))
  fi

  # The sweep-config contract: metric is logged, every axis is read by the
  # trainer, pinned values do not silently override --config, paths exist.
  if "$PY" -m pytest tests/test_sweep_configs.py tests/test_cli_sweep_overrides.py \
       tests/test_run_artifacts.py -q >/tmp/specsr_sweep_tests.log 2>&1; then
    ok "sweep/CLI/artifact tests: $(tail -1 /tmp/specsr_sweep_tests.log)"
  else
    echo "  SWEEP CONFIG TESTS FAILED:"; tail -20 /tmp/specsr_sweep_tests.log
    errs=$((errs+1))
  fi

  grep -qs "api.wandb.ai" "$HOME/.netrc" && ok "wandb credentials" \
    || { echo "  wandb not logged in (run: wandb login)"; errs=$((errs+1)); }

  # Optional: start/finish notifications. Never a reason to refuse to launch --
  # the sweep runs identically either way.
  local notifier; notifier="$(notify_wrapper)"
  if [[ -z "$notifier" ]]; then
    echo "  note: no notifier found; the agent will run unwrapped"
  elif [[ "$notifier" == *notify-run ]] && ! "$notifier" --check | grep -q '^notify-run: configured'; then
    echo "  note: notify-run is not configured, so no mail will be sent."
    echo "        To enable it, see the header of scripts/notify-run."
  else
    ok "notifications via $(basename "$notifier")"
  fi

  # A sweep writes a checkpoint per trial, and running out of disk mid-sweep
  # loses the trials already finished as well as the one in flight.
  local free_gb; free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if [[ "$free_gb" -ge "$MIN_DISK_GB" ]]; then ok "disk free: ${free_gb}G"
  else
    echo "  LOW DISK: ${free_gb}G free (want >= ${MIN_DISK_GB}G)."
    errs=$((errs+1))
  fi

  screen -ls 2>/dev/null | grep -q "$TAG" \
    && { echo "  a screen named '$TAG' already exists -- kill it first"; errs=$((errs+1)); } \
    || ok "no conflicting screen session"

  [[ $errs -eq 0 ]] || fail "$errs preflight check(s) failed. Nothing was created or launched."
  log "Preflight passed"
}

main() {
  preflight
  [[ $PREFLIGHT_ONLY -eq 1 ]] && { log "--preflight given, stopping."; exit 0; }

  if [[ -z "$SWEEP_ID" ]]; then
    log "Creating sweep from configs/sweeps/${STAGE}.yaml"
    local out; out="$("$PY" -m wandb sweep --project "$PROJECT" "$SWEEP_CFG" 2>&1 || true)"
    echo "$out"
    # `wandb sweep` prints the id on its "Run sweep agent with:" line.
    SWEEP_ID="$(echo "$out" | grep -oE 'wandb agent [^ ]+' | tail -1 | awk '{print $3}' || true)"
    [[ -n "$SWEEP_ID" ]] || fail "could not parse a sweep id from wandb output (see above)"
  else
    log "Attaching to existing sweep $SWEEP_ID"
  fi

  log "Launching agent in screen '$TAG'"
  local notifier; notifier="$(notify_wrapper)"
  if [[ -n "$notifier" ]]; then
    screen -dmS "$TAG" "$notifier" "${TAG}_$(date +%Y%m%d)" \
        "$PY" -u -m wandb agent "$SWEEP_ID"
  else
    screen -dmS "$TAG" "$PY" -u -m wandb agent "$SWEEP_ID"
  fi

  sleep 5
  screen -ls | grep -q "$TAG" || fail "screen session '$TAG' did not start"

  cat <<EOF

  sweep     $SWEEP_ID
  dashboard https://wandb.ai/$(echo "$SWEEP_ID" | cut -d/ -f1,2)/sweeps/$(echo "$SWEEP_ID" | rev | cut -d/ -f1 | rev)
  screen    screen -r $TAG      (detach with Ctrl-A D)
  stop      screen -S $TAG -X quit && pkill -f 'wandb agent'
  trials    ls -dt runs/${STAGE}_*

EOF
  log "Agent running."
}

main
