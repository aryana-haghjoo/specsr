#!/usr/bin/env bash
#
# run_all_stages.sh — train SR1 -> ZHead -> SR2 as one chain.
#
#   ./scripts/run_all_stages.sh --smoke     # ~minutes, validates the whole chain
#   ./scripts/run_all_stages.sh             # the real overnight run
#   ./scripts/run_all_stages.sh --preflight # checks only, runs nothing
#   ./scripts/run_all_stages.sh --finetune  # warm-start SR1/SR2 from an existing
#                                           # checkpoint bundle ($SPECSR_INIT_CK)
#                                           # with configs/finetune/*.yaml; the
#                                           # zhead stage always trains fresh
#
# Design notes
# ------------
# * If a `train-notify` command is on PATH, every stage is wrapped in it
#   SEPARATELY, so each emails its own start/finish with its own W&B link;
#   without it each stage runs directly. `python -u` either way, or the link may
#   not be scraped before the start notification is sent.
# * The chain stops at the first failure. A stage that dies at 02:00 must not
#   let the next stage start against a missing checkpoint and waste the night.
# * Preflight runs before anything is launched. Almost every overnight failure
#   is a missing file, a full disk or an unwritable directory -- all knowable in
#   advance, which is the entire point of checking first.
# * --smoke drives the SAME code path with tiny limits via SPECSR_* environment
#   variables, writing to a scratch directory so real checkpoints are untouched.

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
TAG="${SPECSR_TAG:-dr4_groupsplit_$(date +%Y%m%d)}"

SMOKE=0
PREFLIGHT_ONLY=0
FINETUNE=0
# Which stage to start at. A stage that is skipped must still have produced its
# checkpoint on an earlier run -- require_artifact enforces that below, so
# resuming can never train against a stale or missing input.
FROM_STAGE=sr1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)     SMOKE=1 ;;
    --preflight) PREFLIGHT_ONLY=1 ;;
    --finetune)  FINETUNE=1 ;;
    --from)      FROM_STAGE="${2:?--from needs a stage: sr1|zhead|sr2}"; shift ;;
    --from=*)    FROM_STAGE="${1#*=}" ;;
    -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$FROM_STAGE" in
  sr1) START_AT=1 ;;
  zhead) START_AT=2 ;;
  sr2) START_AT=3 ;;
  *) echo "--from must be one of: sr1 zhead sr2 (got '$FROM_STAGE')" >&2; exit 2 ;;
esac

if [[ $SMOKE -eq 1 ]]; then
  OUT_DIR="$REPO/runs/smoke_$(date +%Y%m%d_%H%M%S)"
  export SPECSR_EPOCHS=1
  export SPECSR_LIMIT_TRAIN_BATCHES=3
  export SPECSR_LIMIT_VAL_BATCHES=2
  export SPECSR_WANDB_MODE=offline
  TAG="smoke_$(date +%H%M%S)"
else
  OUT_DIR="${SPECSR_OUT_DIR:-$REPO/runs/$TAG}"
fi
export SPECSR_OUT_DIR="$OUT_DIR"
export SPECSR_DATASET="$DATASET"

# Every stage writes into $OUT_DIR, never into the source tree. The old layout
# wrote checkpoints beside the training scripts, so a rerun silently overwrote
# the previous model in place.
SR1_DIR="$OUT_DIR/sr1"
ZH_DIR="$OUT_DIR/zhead"
SR2_DIR="$OUT_DIR/sr2"

SR1_CKPT="$SR1_DIR/best_superres_model.pth"
# `specsr train zhead` writes the canonical name since 2026-08-14; it used to
# suffix the arm (`best_zhead_sr1.pth`), which is why the bundle step below had
# to rename it. Older run directories still carry the suffixed name, so resume
# runs (`--from sr2`) accept either.
ZH_CKPT="$ZH_DIR/best_zhead.pth"
[[ -f "$ZH_CKPT" ]] || [[ ! -f "$ZH_DIR/best_zhead_sr1.pth" ]] || ZH_CKPT="$ZH_DIR/best_zhead_sr1.pth"

# Fine-tune mode swaps in the configs that mirror the init checkpoints'
# architecture and passes --init-ckpt to the SR1 and SR2 stages. The zhead
# stage never warm-starts: its v2 architecture is incompatible with v1
# checkpoints, and it is cheap to train fresh.
INIT_CK="${SPECSR_INIT_CK:-$REPO/checkpoints/checkpoints_run5_20260728}"
if [[ $FINETUNE -eq 1 ]]; then
  SR1_CFG="$REPO/configs/finetune/sr1.yaml"
  SR2_CFG="$REPO/configs/finetune/sr2.yaml"
  SR1_INIT_ARGS=(--init-ckpt "$INIT_CK/best_superres_model.pth")
  SR2_INIT_ARGS=(--init-ckpt "$INIT_CK/best_sr2.pth")
else
  SR1_CFG="$REPO/configs/sr1.yaml"
  SR2_CFG="$REPO/configs/sr2.yaml"
  SR1_INIT_ARGS=()
  SR2_INIT_ARGS=()
fi

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

# --smoke runs the REAL training scripts, and their checkpoint paths are
# hardcoded -- so a 1-epoch smoke model lands on top of the good checkpoint and
# destroys it. That happened: a smoke run replaced a 10-hour SR1 and a 5-hour
# ZHead, and they had to be pulled back out of W&B. Stash the checkpoints and
# put them back on any exit, so validating the chain can never cost a run.
SMOKE_STASH=""
# Checkpoint safety is now structural, not a trap.
#
# A smoke run once destroyed a 10-hour SR1 and a 5-hour ZHead: the training
# scripts saved to their own directory, so `--smoke` overwrote the real weights
# in place and the only recovery was a W&B download. That was patched with a
# stash/restore trap, which worked but depended on remembering to arm it.
#
# Every stage now takes --out-dir and a smoke run gets $REPO/runs/smoke_<stamp>,
# so a rehearsal cannot address the same path as a real run. Nothing to stash.

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight() {
  log "Preflight"
  local errs=0

  [[ -x "$PY" ]] && ok "python: $PY" || { echo "  MISSING python: $PY"; errs=$((errs+1)); }

  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ok "CUDA: $("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))')"
  else
    echo "  NO CUDA -- training would fall back to CPU and never finish overnight"; errs=$((errs+1))
  fi

  if "$PY" -c "import specsr, wandb, numpy, yaml" 2>/dev/null; then
    ok "imports: specsr, wandb, numpy, yaml"
  else
    echo "  import failure:"; "$PY" -c "import specsr, wandb, numpy, yaml" || true; errs=$((errs+1))
  fi

  [[ -f "$DATASET" ]] && ok "dataset: $DATASET ($(du -h "$DATASET" | cut -f1))" \
    || { echo "  MISSING dataset: $DATASET"; errs=$((errs+1)); }

  [[ -f "$SR1_CFG" ]] && ok "sr1 config: $SR1_CFG" \
    || { echo "  MISSING sr1 config: $SR1_CFG"; errs=$((errs+1)); }

  if [[ $FINETUNE -eq 1 ]]; then
    for f in "$INIT_CK/best_superres_model.pth" "$INIT_CK/best_sr2.pth"; do
      [[ -f "$f" ]] && ok "finetune init: $f" \
        || { echo "  MISSING finetune init checkpoint: $f"; errs=$((errs+1)); }
    done
  fi

  # Import every stage script. Each guards its entry point behind
  # `if __name__ == "__main__"`, so importing runs no training but does resolve
  # every import. A moved module breaks a stage at import time, which otherwise
  # only surfaces when the chain reaches it -- SR2 died this way on a module the
  # refactor had relocated, two stages deep.
  if "$PY" - <<'EOF'
import importlib, traceback
# Import as package modules, not by file path: they use relative imports, so
# loading them as standalone files fails on `from ..data import ...` and would
# report a spurious failure.
bad = []
for name in ("sr1", "zhead", "sr2"):
    try:
        importlib.import_module(f"specsr.training.{name}")
    except Exception:
        bad.append(name)
        print(f"  stage '{name}' fails at import:")
        traceback.print_exc(limit=3)
# The CLI is how the chain actually invokes them, so check its dispatch too.
for mod in ("specsr.training.runner", "specsr.inference.pipeline",
            "specsr.evaluation.runner", "specsr.data.build"):
    try:
        importlib.import_module(mod)
    except Exception:
        bad.append(mod)
        print(f"  CLI target '{mod}' fails at import:")
        traceback.print_exc(limit=3)
raise SystemExit(1 if bad else 0)
EOF
  then ok "all stages and CLI dispatch targets import cleanly"
  else echo "  STAGE IMPORT FAILURE"; errs=$((errs+1)); fi

  # The split must be leak-free before we spend a night on it.
  if [[ -f "$DATASET" ]]; then
    if "$PY" - "$DATASET" <<'EOF'
import sys, numpy as np
from specsr.data.splits import get_or_make_split_3way, assert_clean_3way
import numpy as np
tr, va, te, _ = get_or_make_split_3way(sys.argv[1])
d = np.load(sys.argv[1], allow_pickle=True)
g, orig = np.asarray(d["parent_id"]), np.asarray(d["is_original"], bool)
assert_clean_3way(tr, va, te, g, orig)
n = lambda i: len(np.unique(g[i]))
print(f"  ok   split: {n(tr)} train / {n(va)} val / {n(te)} test galaxies, 0 leaked; "
      f"val+test are originals only")
EOF
    then :; else echo "  SPLIT CHECK FAILED"; errs=$((errs+1)); fi
  fi

  # Optional: start/finish notifications. Never a reason to refuse to launch --
  # the chain runs identically either way.
  local notifier; notifier="$(notify_wrapper)"
  if [[ -z "$notifier" ]]; then
    echo "  note: no notifier found; stages will run unwrapped"
  elif [[ "$notifier" == *notify-run ]] && ! "$notifier" --check | grep -q '^notify-run: configured'; then
    echo "  note: notify-run is not configured, so no mail will be sent."
    echo "        To enable it, see the header of scripts/notify-run."
  else
    ok "notifications via $(basename "$notifier")"
  fi

  if [[ $SMOKE -eq 0 ]]; then
    grep -qs "api.wandb.ai" "$HOME/.netrc" && ok "wandb credentials" \
      || { echo "  wandb not logged in (run: wandb login)"; errs=$((errs+1)); }
  fi

  mkdir -p "$OUT_DIR" && [[ -w "$OUT_DIR" ]] && ok "output dir writable: $OUT_DIR" \
    || { echo "  output dir not writable: $OUT_DIR"; errs=$((errs+1)); }

  local free_gb; free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if [[ "$free_gb" -ge 20 ]]; then ok "disk free: ${free_gb}G"
  else echo "  LOW DISK: ${free_gb}G free (want >= 20G)"; errs=$((errs+1)); fi

  local free_mb; free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [[ "$free_mb" -ge 8000 ]]; then ok "GPU free: ${free_mb} MiB"
  else echo "  GPU busy: only ${free_mb} MiB free -- another job may be running"; errs=$((errs+1)); fi

  [[ $errs -eq 0 ]] || fail "$errs preflight check(s) failed. Nothing was launched."
  log "Preflight passed"
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
run_stage() {
  local name="$1" dir="$2"; shift 2
  local stage_log="$OUT_DIR/${name}.log"
  log "Stage: $name  (log: $stage_log)"

  # tee so the chain can inspect what the stage reported while the user still
  # sees live progress.
  # `if`, not `&&` -- under `set -e` a failing `&&` list at statement level
  # exits the script, so the absent-notifier case would kill the chain.
  local -a wrapped=("$@")
  local notifier; notifier="$(notify_wrapper)"
  if [[ -n "$notifier" ]]; then
    wrapped=("$notifier" "${TAG}_${name}" "$@")
  fi
  ( cd "$dir" && "${wrapped[@]}" ) 2>&1 | tee "$stage_log" \
    || fail "stage '$name' failed -- chain stopped, later stages not started"
  # tee swallows the exit status, so consult PIPESTATUS explicitly.
  [[ "${PIPESTATUS[0]}" -eq 0 ]] \
    || fail "stage '$name' failed -- chain stopped, later stages not started"

  assert_dataset_used "$name" "$stage_log"
  log "Stage $name finished"
}

# A stage silently training on the wrong product is invisible in the metrics and
# expensive to discover late -- a smoke run caught exactly that, with preflight
# validating DR4 while SR1 trained on DR3 from a hardcoded path.
assert_dataset_used() {
  local name="$1" stage_log="$2"
  local want used
  want="$(readlink -f "$DATASET")"
  # `|| true` is essential: grep exits 1 when a stage does not print the line,
  # and under `set -e` a failing command substitution kills the script silently,
  # with no error message at all.
  used="$(grep -m1 '^\[specsr\] dataset: ' "$stage_log" 2>/dev/null | sed 's/^\[specsr\] dataset: //' || true)"
  if [[ -z "$used" ]]; then
    echo "  warn: stage '$name' did not report a dataset path" >&2
    return 0
  fi
  used="$(readlink -f "$used")"
  [[ "$used" == "$want" ]] \
    || fail "stage '$name' used the WRONG dataset
    expected: $want
    actual:   $used"
  ok "stage '$name' confirmed on $(basename "$used")"
}

require_artifact() {
  [[ -f "$1" ]] || fail "expected artifact missing after its stage: $1"
  ok "artifact: $1 ($(du -h "$1" | cut -f1))"
  check_checkpoint_sane "$1"
}

# "The file exists" is a weak success signal: a diverged run writes a checkpoint
# full of NaNs, and a mis-wired one writes all zeros. Both would sail through an
# existence check and only surface hours later.
check_checkpoint_sane() {
  "$PY" - "$1" <<'EOF' || fail "checkpoint failed its sanity check: $1"
import sys, torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)

sd = ck
for key in ("model_state_dict", "state_dict", "sr2_state_dict"):
    if isinstance(ck, dict) and key in ck:
        sd = ck[key]
        break
if not (isinstance(sd, dict) and any(torch.is_tensor(v) for v in sd.values())):
    sd = next(v for v in ck.values()
              if isinstance(v, dict) and any(torch.is_tensor(t) for t in v.values()))

tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
nonfinite = [k for k, v in tensors.items() if not torch.isfinite(v).all()]
allzero = [k for k, v in tensors.items()
           if v.numel() > 1 and float(v.abs().max()) == 0.0]

# Some heads are deliberately zero-initialised (SR2's CNN output, for one), so a
# few all-zero tensors are expected; everything being zero is not.
if nonfinite:
    print(f"  NON-FINITE weights in {len(nonfinite)} tensor(s): {nonfinite[:5]}")
    sys.exit(1)
if tensors and len(allzero) == len(tensors):
    print("  every tensor is all-zero -- the model did not train")
    sys.exit(1)
print(f"  ok   sane: {len(tensors)} tensors, 0 non-finite, {len(allzero)} all-zero")
EOF
}

main() {
  preflight
  [[ $PREFLIGHT_ONLY -eq 1 ]] && { log "--preflight given, stopping."; exit 0; }

  if [[ $SMOKE -eq 1 ]]; then
    log "SMOKE RUN: 1 epoch, 3 train / 2 val batches, wandb offline, out=$OUT_DIR"
  fi

  local t0=$SECONDS

  # Plain `[[ ... ]] && cmd` is wrong here: when the test is false the compound
  # returns 1 and `set -e` kills the chain. Use if/fi.
  if [[ $START_AT -le 1 ]]; then
    run_stage sr1 "$REPO" "$PY" -u -m specsr.cli train sr1 \
        --config "$SR1_CFG" --dataset "$DATASET" --out-dir "$SR1_DIR" \
        ${SR1_INIT_ARGS[@]+"${SR1_INIT_ARGS[@]}"}
  else
    log "Skipping sr1 (--from $FROM_STAGE); reusing its checkpoint"
  fi
  require_artifact "$SR1_CKPT"

  # The chain trains the `sr1` arm, which is the head SR2 is conditioned on.
  # The lowres/hires/sr2 arms of the redshift comparison are separate runs --
  # see `specsr train zhead --source`.
  if [[ $START_AT -le 2 ]]; then
    run_stage zhead "$REPO" "$PY" -u -m specsr.cli train zhead --source sr1 \
        --config "$REPO/configs/zhead.yaml" --dataset "$DATASET" \
        --out-dir "$ZH_DIR" \
        --sr1-config "$SR1_CFG" --sr1-ckpt "$SR1_CKPT"
  else
    log "Skipping zhead (--from $FROM_STAGE); reusing its checkpoint"
  fi
  require_artifact "$ZH_CKPT"

  run_stage sr2 "$REPO" "$PY" -u -m specsr.cli train sr2 \
      --config "$SR2_CFG" --dataset "$DATASET" \
      --out-dir "$SR2_DIR" \
      --sr1-config "$SR1_CFG" --sr1-ckpt "$SR1_CKPT" --zhead-ckpt "$ZH_CKPT" \
      ${SR2_INIT_ARGS[@]+"${SR2_INIT_ARGS[@]}"}
  require_artifact "$SR2_DIR/best_sr2.pth"

  # Assemble the four files downstream consumers expect (train_zhead_arms.sh,
  # `specsr infer --checkpoints`) under one directory, in their expected names.
  local bundle="$OUT_DIR/checkpoints_bundle"
  mkdir -p "$bundle"
  cp "$SR1_CKPT" "$bundle/best_superres_model.pth"
  cp "$SR1_CFG" "$bundle/config_logR.yaml"
  cp "$ZH_CKPT" "$bundle/best_zhead.pth"
  cp "$SR2_DIR/best_sr2.pth" "$bundle/best_sr2.pth"
  (cd "$bundle" && md5sum *.pth *.yaml > MD5SUMS)
  log "Checkpoint bundle -> $bundle"

  log "Chain complete (from $FROM_STAGE) in $(( (SECONDS - t0) / 60 )) min"
  [[ $SMOKE -eq 1 ]] && log "Smoke run OK -- the chain is safe to launch overnight."
  return 0
}

main
