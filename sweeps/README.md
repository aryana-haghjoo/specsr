# Sweep trial tables

One row per trial of the three hyperparameter sweeps reported in Appendix A of
the paper, exported from Weights & Biases:

| file | stage | W&B sweep | trials |
|---|---|---|---|
| `sweep_sr1_c9tct37z.csv` | SR1 | [`c9tct37z`](https://wandb.ai/AI-Astro/spectral-superresolution/sweeps/c9tct37z) | 40 |
| `sweep_zhead_kdtc1kr5.csv` | redshift head | [`kdtc1kr5`](https://wandb.ai/AI-Astro/spectral-superresolution/sweeps/kdtc1kr5) | 42 |
| `sweep_sr2_y2zvwvkf.csv` | SR2 | [`y2zvwvkf`](https://wandb.ai/AI-Astro/spectral-superresolution/sweeps/y2zvwvkf) | 40 |

These are **tracked**, unlike the regenerable artifacts under `cache/`, because
the paper's Appendix A figure is drawn from them and has to be reproducible from
a clean checkout without W&B credentials:

    python scripts/make_sweep_figure.py            # draw from these tables
    python scripts/make_sweep_figure.py --refresh  # re-export them, then draw

The figure lands in the package output directory (`SPECSR_OUTPUT_DIR`, default
`./outputs/figures`); pass `--out <stem>` to write it somewhere else.

Each metric is stored twice, at the first hyperband rung (`*_at_rung1`, the last
epoch every trial reached) and at the trial's own last epoch (`*_final`);
`last_epoch` and `full_length` record where hyperband stopped it. The two columns
exist because a trial's final value is reported at whichever of three budgets it
survived to, so ranking trials on it alone confounds a bad configuration with an
early-stopped one.

Columns after those are the searched hyperparameters of that sweep, named as in
`configs/sweeps/<stage>.yaml`.
