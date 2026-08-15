# Scripts — regenerating the published figures

Thin entry points. All the logic is in `specsr`: models and data in
`specsr.models` / `specsr.data`, figure functions in `specsr.plotting`,
statistics in `specsr.metrics` and `specsr.linefit`, and the split/inference
plumbing in `specsr.evaluation`. These files do argument parsing and file IO.

## After a new SR2 lands

```bash
CK=runs/<tag>/checkpoints_bundle   # assembled by run_all_stages.sh
python scripts/make_predictions.py --sr2-ckpt "$CK/best_sr2.pth" \
    --sr1-ckpt "$CK/best_superres_model.pth" --sr1-config "$CK/config_logR.yaml" \
    --zhead-ckpt "$CK/best_zhead.pth"
python scripts/flux_conservation.py --split val --ckpt "$CK/best_sr2.pth"
python scripts/make_figures.py          # all figures -> $SPECSR_OUTPUT_DIR/figures
```

`finetune_and_zarms.sh` automates the front half — fine-tune chain, cache
rebuild, redshift-comparison arms — as one screen launch; see
`docs/guides/training.md`.

`make_figures.py --only <name>` rebuilds a subset. `--only coverage` needs
`SPECSR_JADES_ROOT` set, since it reads native grating spectra from the raw tree
rather than the built product.

`make_predictions.py` caches one `.npz` holding LR / SR1 / SR2 / HR spectra in
physical units, their uncertainties, redshifts, the wavelength grid and row
provenance. Figures read that cache instead of re-running the models, so every
panel in the paper comes from provably the same predictions.

## The test split is sealed

`--split test` requires `--allow-test` and prints a banner. Use it **once**, on a
frozen model, for the numbers that go in the paper. Everything diagnostic runs on
`val`. A held-out set that gets looked at repeatedly is a validation set with a
misleading name — which is how the augmentation leak stopped meaning anything.

## Defaults point at a frozen archive, never at a live run directory

A run directory is overwritten in place by a running chain, so evaluating
against one mid-run silently compares SR2 checkpoints sitting on top of
*different* SR1s. Defaults therefore point at a frozen checkpoint archive; pass
explicit paths for anything else. Every script echoes the three checkpoint paths
it actually loaded.

## MSE is not space-invariant — say which one you mean

| space | SR1 | SR2 |
|---|---|---|
| per-spectrum normalised | 0.845963 | 0.864069 (SR2 **worse** 2.1%) |
| physical HR units | 1.2196e-38 | 1.2175e-38 (SR2 **better** 0.17%) |

Normalising divides each spectrum by its own σ, which weights faint objects
equally with bright ones; physical units let bright objects dominate. SR2 is
better on bright spectra and worse on faint ones, so the comparison changes sign
with the choice of space. Quote one, state which, and do not switch between them
between sections.

## Figures and their sources

| figure | `--only` | function |
|---|---|---|
| `generated_spectra.png` | `spectrum` | `plotting.plot_spectra_with_inset` |
| `residual_maps.png` | `residual_maps` | `plotting.plot_residual_maps` |
| `psd.png` | `psd` | `plotting.plot_residual_psd` |
| `sn_comparison.png` | `snr` | `plotting.plot_snr_comparison` + `linefit` |
| `augmentation.png` | `augmentation` | `plotting.plot_augmentation_family` |
| `matched_spectra_comparison.png` | `coverage` | `plotting.plot_disperser_coverage` |
| `line_flux_comparison.png` | `line_flux` | `plotting.plot_line_flux_comparison` |
| `fig_architecture.pdf` | — | `make_architecture_figure.py` (own entry point) |
| `sweep.png` | — | W&B export; needs a fresh sweep, not reproducible here |

`ablate_conditioning_head.py` answers "which redshift head should condition
SR2?" by swapping the head under the frozen released SR2 — no training, minutes
to run. It is a *lower* bound on the benefit, since SR2 was trained against the
SR1 head. Its second table is the useful one: on bright lines the two heads
almost always agree, and ~19% of bright [O III] lines are missing from SR2's
output even when the redshift is right, which is an amplitude problem rather than
a placement one.

The architecture schematic is the one figure not built by `make_figures.py`,
because it needs the loaded models rather than the prediction cache:

```bash
python scripts/make_architecture_figure.py   # -> $SPECSR_OUTPUT_DIR/figures/fig_architecture.pdf
```

Every number it prints — channel widths, block counts, token count, window size,
redshift-bin count, per-stage parameter totals — is read off the released
checkpoints at draw time, and the three inset panels are real predictions for a
held-out galaxy chosen by the same `rank_doublet_examples` criterion as the
example-spectrum figure. Nothing on it is transcribed by hand, so a retrain that
changes a hyperparameter changes the diagram instead of silently invalidating it.
This replaced a TikZ drawing that had drifted: it still advertised teacher
forcing, a Gaussian redshift head and a two-channel SR1 input, none of which
survive in the code.

The script asserts that the prediction cache was built from the same three
checkpoints it loads, and exits rather than illustrating a model the paper does
not measure.

`line_flux_comparison.png` exists because the S/N figure never references the HR
truth, so it cannot show the lines are *correct* — only that something was
detected confidently.

Every paper figure is drawn under `plotting.PAPER_RC` (DejaVu Serif, with
`mathtext.fontset` matched to it). The manuscript is set in a serif face, and a
sans-serif panel dropped into it is immediately visible on the page. Wrap new
figures in `plt.rc_context(PAPER_RC)` rather than restating the family locally;
setting `font.family` without `mathtext.fontset` is the usual miss, and leaves
every `$\lambda$` in a different face from the words around it.

The example spectra are **selected, not hard-coded**: `metrics.rank_doublet_examples`
picks objects where the [O III] doublet is blended in LR and resolved in SR, and
checks the SR peaks land at the right wavelengths — SR2 conditions its line
branch on the predicted redshift, so a bad redshift yields a crisp doublet in the
wrong place, which ranking on separation alone would happily select.

## Watch the absolute constants

Seven hardcoded epsilons inherited from notebook code were tuned for normalised
flux (O(1)) and are silently wrong on physical spectra (~1e-21): they set the
answer instead of guarding against zero, producing flat lines, empty panels and
S/N of exactly zero. All are now relative. If you port more plotting code, check
every literal.
