"""The figure cache must decode redshift the same way training does.

Regression test for a silent bug that invalidated every cached figure. The
cache builder in :mod:`specsr.evaluation` hand-rolled the redshift decode

    mu_n = z_min_n + (z_max_n - z_min_n) * sigmoid(mu_raw)

instead of calling :meth:`RedshiftTransform.predict` with
``bounded=zhead.bounded_mean``. That squash is correct for the Gaussian head,
whose output is unbounded, and *wrong* for the classification head, whose mean
is already in normalised redshift units. Squashing it a second time compressed
every estimate towards the range centre.

Nothing raised. The cache came out with a median ``|dz|/(1+z)`` of 0.96 rather
than 0.0018, predicting roughly twice each galaxy's true redshift, and because
``zhat`` also drives the line mask and SR2's redshift channel, SR2 was evaluated
at a redshift it was never given at training time. Every figure built from the
cache -- example spectra, residual maps, PSD, line S/N, line flux -- inherited
that. The values stayed finite, ordered and physically plausible throughout,
which is why it survived so long.

Every other consumer already branched on ``zhead.bounded_mean``; the evaluation
path was the one hand-rolled copy. These tests assert on numbers, because the
failure mode produces no exception.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "paired_DR4_logR.npz"
SR1 = REPO / "runs/finetune_20260730_003724/sr1/best_superres_model.pth"
SR1_CONFIG = REPO / "checkpoints/checkpoints_baseline_20260726/config_logR.yaml"
ZHEAD = REPO / "runs/zhead_pdf_8020/sr1/best_zhead_sr1.pth"

# The released SR1-fed head reports med |dz|/(1+z) = 0.0018 on this split. The
# double-squash gave 0.96. Anything above this threshold means the decode is
# wrong again, not that the model drifted.
MAX_MED_ABS_DZ = 0.05

needs_artifacts = pytest.mark.skipif(
    not (DATASET.exists() and SR1.exists() and ZHEAD.exists() and SR1_CONFIG.exists()),
    reason="dataset or released checkpoints not available",
)


@pytest.fixture(scope="module")
def cached():
    """Run the cache builder once; both artifact tests read the same result."""
    from specsr.evaluation import load_pipeline, predict

    p = load_pipeline(sr2_ckpt=None, sr1_ckpt=str(SR1), sr1_config=str(SR1_CONFIG),
                      zhead_ckpt=str(ZHEAD), dataset=str(DATASET))
    return predict(p, "val", dataset=str(DATASET))


def test_classification_head_mean_is_not_squashed_again():
    """The transform contract, isolated from any checkpoint.

    ``bounded=False`` must pass the estimate through untouched. With
    ``bounded=True`` the same value is pulled towards the range centre, which is
    the arithmetic that produced the bad cache.
    """
    from specsr.training.ztransform import RedshiftTransform

    ztf = RedshiftTransform(mean=3.3268, std=1.8668, z_min_n=-1.782, z_max_n=6.246)
    # A galaxy at z = 3.353, i.e. essentially at the mean of the transform.
    z_true = 3.353
    mu_n = torch.tensor([(z_true - ztf.mean) / ztf.std])
    log_var = torch.tensor([-6.0])

    z_ok, _ = ztf.predict(mu_n, log_var, bounded=False)
    z_bad, _ = ztf.predict(mu_n, log_var, bounded=True)

    assert float(z_ok) == pytest.approx(z_true, abs=1e-4)
    # The observed failure: z ~ 3.35 came back as ~7.5.
    assert float(z_bad) > 7.0, "the double-squash should be dramatic, not subtle"


@needs_artifacts
def test_prediction_cache_recovers_redshift(cached):
    """The real regression: run the cache builder and check the redshifts.

    This exercises :func:`specsr.evaluation.predict` itself rather than the
    transform, because the bug was in the call site, not the helper.
    """
    z_true = np.asarray(cached["z_true"], dtype=float)
    z_pred = np.asarray(cached["z_pred"], dtype=float)
    med = float(np.median(np.abs(z_pred - z_true) / (1.0 + z_true)))

    assert med < MAX_MED_ABS_DZ, (
        f"median |dz|/(1+z) = {med:.4f} exceeds {MAX_MED_ABS_DZ}. The cache is "
        "decoding redshift wrongly -- check that predict() passes "
        "bounded=zhead.bounded_mean rather than always applying the sigmoid "
        "range-squash."
    )


@needs_artifacts
def test_cache_and_inference_pipeline_agree_on_redshift(cached):
    """Two decode paths, one answer.

    The bug existed because the same decode was written twice. If they ever
    disagree again, one of them is wrong and this says so.
    """
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(
        sr1_ckpt=str(SR1), sr1_config=str(SR1_CONFIG), zhead_ckpt=str(ZHEAD),
        dataset=str(DATASET))
    with np.load(DATASET, allow_pickle=True) as d:
        rows = np.asarray(cached["row_index"], dtype=int)[:32]
        flux_low = np.asarray(d["flux_low"][rows], dtype=np.float32)

    z_pipe = np.asarray(pipe(flux_low).z, dtype=float)
    z_cache = np.asarray(cached["z_pred"], dtype=float)[:32]

    assert np.allclose(z_pipe, z_cache, rtol=1e-3, atol=1e-3), (
        "the cache builder and SpecSRPipeline decode redshift differently; "
        "they must share RedshiftTransform.predict"
    )


def _doublet(wave, z, amp4959, amp5007, sigma=0.0012, continuum=1.0):
    """A clean two-Gaussian [O III] doublet on a flat continuum."""
    y = np.full_like(wave, continuum)
    for lam, a in ((0.4959, amp4959), (0.5007, amp5007)):
        y = y + a * np.exp(-0.5 * ((wave - lam * (1 + z)) / sigma) ** 2)
    return y


def test_doublet_ranking_rejects_over_emitted_examples():
    """The example figure must not showcase the model overshooting.

    Regression test for the paper's qualitative figure. The ranking scored on
    the *SR* peak height, so emitting harder scored better. On the released
    model the top-ranked example over-emitted [O III] 5007 by 18.8x -- the
    99.7th percentile of the held-out set, against a median of 0.55 -- and the
    HR reference rendered as a flat line beside it.

    Candidates 0 and 1 are identical except in SR amplitude: one faithful, one
    over-emitting 15x. The faithful one must win, and the over-emitter must be
    rejected outright rather than merely ranked lower. Candidate 2 is a dim
    decoy, present only so the brightness percentile cut has something to
    discard -- with two equally bright candidates a strict ``>`` against the
    0th percentile would exclude both.
    """
    from specsr.metrics import rank_doublet_examples

    wave = np.linspace(1.0, 2.0, 4000)
    z = 2.2
    hr = np.stack([
        _doublet(wave, z, 5.0, 15.0),
        _doublet(wave, z, 5.0, 15.0),
        _doublet(wave, z, 0.5, 1.5),         # dim decoy
    ])
    # LR: same flux, far too broad to separate -- a single blended bump.
    lr = np.stack([_doublet(wave, z, 5.0, 15.0, sigma=0.012)] * 3)
    sr = np.stack([
        _doublet(wave, z, 4.5, 14.0),        # faithful
        _doublet(wave, z, 75.0, 225.0),      # over-emits 15x
        _doublet(wave, z, 0.45, 1.4),        # dim decoy, faithful
    ])
    z_arr = np.full(3, z)
    z_pred = np.full(3, z)

    ranked = rank_doublet_examples(wave, lr, sr, hr, z_arr, z_pred=z_pred,
                                   amp_percentile=0.0)

    assert len(ranked) >= 1, "the faithful example should still be selectable"
    assert ranked[0] == 0, "the faithful example must outrank the over-emitter"
    assert 1 not in ranked, "a 15x over-emitter must be rejected, not just deranked"
