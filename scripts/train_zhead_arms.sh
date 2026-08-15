#!/usr/bin/env bash
# Train the three arms of the paper's redshift comparison, then build the figure.
#
#   screen -dmS zarms ./scripts/train_zhead_arms.sh
#
# One script, one `--source` flag per arm, so architecture, loss, optimiser,
# split and seed are identical by construction. That is the whole basis of the
# comparison: the arms must differ only in what spectrum the head is shown, or
# the result measures tuning effort rather than information content.
#
# Runs sequentially -- one GPU. Each arm is wrapped in a notifier when one is
# available, so each reports its own start and finish.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SPECSR_PYTHON:-}" && -x "$REPO/sup_res/bin/python" ]]; then
  PY="$REPO/sup_res/bin/python"
else
  PY="${SPECSR_PYTHON:-$(command -v python3 || command -v python)}"
fi
# Which frozen chain the sr2 arm is built on. A fine-tune run's bundle is
# passed as ZARMS_CK=runs/<tag>/checkpoints_bundle.
CK="${ZARMS_CK:-$REPO/checkpoints/release}"
DATA="$REPO/data/paired_DR4_logR.npz"
TAG="${TAG:-zarms_$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO/runs/$TAG"
mkdir -p "$OUT"

cd "$REPO"
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

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

for f in "$DATA" "$CK/best_superres_model.pth" "$CK/config_logR.yaml" \
         "$CK/best_zhead.pth" "$CK/best_sr2.pth"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

# lowres and hires see raw spectra, so they need no upstream checkpoints. sr2
# needs the frozen chain, plus the bootstrap head that conditions SR2 -- that
# head is NOT the one being trained, and passing the wrong one would silently
# leak a better redshift into the input.
run_arm() {
  local src="$1"; shift
  log "arm: $src"
  # `if`, not `&&` -- under `set -e` a failing `&&` list at statement level
  # would exit the script when no notifier is installed.
  local -a notify=()
  local notifier; notifier="$(notify_wrapper)"
  if [[ -n "$notifier" ]]; then
    notify=("$notifier" "${TAG}_${src}")
  fi
  "${notify[@]}" \
    "$PY" -u -m specsr.cli train zhead --source "$src" \
        --config "$REPO/configs/zhead.yaml" \
        --dataset "$DATA" \
        --out-dir "$OUT/$src" "$@"
  [[ -f "$OUT/$src/predictions_${src}.npz" ]] \
    || { echo "arm $src produced no predictions" >&2; exit 1; }
}

run_arm lowres
run_arm hires
run_arm sr2 \
  --sr1-ckpt "$CK/best_superres_model.pth" \
  --sr1-config "$CK/config_logR.yaml" \
  --sr2-ckpt "$CK/best_sr2.pth" \
  --zhead-bootstrap-ckpt "$CK/best_zhead.pth"

# The figure is written into the run directory, never straight into the paper.
# A rehearsal that overwrites the paper's figure with 1-epoch noise is the same
# failure as a smoke run overwriting real weights, and harder to notice: the
# file looks like a figure.
log "building the figure"
"$PY" -m specsr.cli evaluate redshift \
    --z-lowres "$OUT/lowres/predictions_lowres.npz" \
    --z-hires  "$OUT/hires/predictions_hires.npz" \
    --z-sr2    "$OUT/sr2/predictions_sr2.npz" \
    --outdir   "$OUT"

# Published figures go to the package output directory. Point SPECSR_OUTPUT_DIR
# at a manuscript's plots directory to regenerate a paper figure in place.
FIGDIR="${SPECSR_OUTPUT_DIR:-$REPO/outputs}/figures"
if [[ -n "${SPECSR_LIMIT_TRAIN_BATCHES:-}${SPECSR_EPOCHS:-}" ]]; then
  log "SMOKE: figure left in $OUT, $FIGDIR untouched"
else
  mkdir -p "$FIGDIR"
  cp "$OUT/redshift_comparison.png" "$FIGDIR/redshift_comparison.png"
  log "done -> $FIGDIR/redshift_comparison.png"
fi
