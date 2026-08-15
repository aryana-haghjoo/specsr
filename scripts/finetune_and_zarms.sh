#!/usr/bin/env bash
# The full validation-retrain sequence, in one launch:
#
#   1. run_all_stages.sh --finetune   SR1 (warm-started) -> ZHead v2 -> SR2
#                                     (warm-started), bundle assembled at the
#                                     end under runs/<tag>/checkpoints_bundle
#   2. make_predictions.py            rebuild the prediction cache from the NEW
#                                     chain; a stale cache would silently mix
#                                     predictions from two different models
#   3. train_zhead_arms.sh            the three-arm redshift comparison and its
#                                     figure, on the new frozen chain
#
#   screen -dmS finetune ./scripts/finetune_and_zarms.sh
#
# Every training stage inside 1. and 3. wraps itself in a notifier when one is
# available, so each reports its own start and finish. See scripts/notify-run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ -z "${SPECSR_PYTHON:-}" && -x "$REPO/sup_res/bin/python" ]]; then
  PY="$REPO/sup_res/bin/python"
else
  PY="${SPECSR_PYTHON:-$(command -v python3 || command -v python)}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_TAG="${SPECSR_TAG:-finetune_${STAMP}}"
BUNDLE="$REPO/runs/$CHAIN_TAG/checkpoints_bundle"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

# SPECSR_FROM=sr2 (with SPECSR_TAG pointing at an existing run) resumes a chain
# whose earlier stages already produced checkpoints, rather than paying for them
# twice. run_all_stages.sh still verifies every reused artefact before using it.
FROM_ARGS=()
[[ -n "${SPECSR_FROM:-}" ]] && FROM_ARGS=(--from "${SPECSR_FROM}")

log "1/3 fine-tune chain -> runs/$CHAIN_TAG ${SPECSR_FROM:+(resuming from ${SPECSR_FROM})}"
SPECSR_TAG="$CHAIN_TAG" ./scripts/run_all_stages.sh --finetune ${FROM_ARGS[@]+"${FROM_ARGS[@]}"}

log "2/3 rebuilding prediction cache from $BUNDLE"
"$PY" scripts/make_predictions.py \
    --sr1-ckpt "$BUNDLE/best_superres_model.pth" \
    --sr1-config "$BUNDLE/config_logR.yaml" \
    --zhead-ckpt "$BUNDLE/best_zhead.pth" \
    --sr2-ckpt "$BUNDLE/best_sr2.pth"

log "3/3 redshift comparison arms on the new chain"
ZARMS_CK="$BUNDLE" TAG="zarms_v2_${STAMP}" ./scripts/train_zhead_arms.sh

log "all done: chain runs/$CHAIN_TAG, arms runs/zarms_v2_${STAMP}"
