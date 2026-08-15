# Training

The pipeline is trained in three sequential stages. Each stage consumes the
frozen checkpoint of the previous one:

1. **SR1** — super-resolution backbone
2. **ZHead** — redshift inference from the SR1 output (SR1 frozen)
3. **SR2** — physics-informed residual refiner (SR1 and ZHead frozen)

## Launching runs

Every stage is one CLI call, and every upstream checkpoint is named explicitly
rather than discovered, so a trained model records what it sits on top of:

```bash
python -u -m specsr.cli train sr1 \
    --config configs/sr1.yaml --out-dir runs/sr1
```

Long runs should be launched inside a `screen` session so they survive logoff.

### Notifications are opt-in

The launch scripts can email you when a stage starts and when it finishes. This
is **off unless you configure it with your own address**, and the scripts behave
identically either way. They look for a notifier in this order:

1. `$SPECSR_NOTIFY_CMD`, if you point it at a wrapper of your own
2. `train-notify` on `PATH`, for an existing setup
3. `scripts/notify-run`, which ships with the repository

`notify-run` unconfigured is a transparent passthrough — same output, same exit
code, no mail, no delay — which is why the scripts can wrap every stage without
imposing anything on someone who never asked for email. Configuring it is a
single file, `~/.specsr_notify.conf`, holding your address and SMTP details; see
the [installation guide](installation.md) or run
`./scripts/notify-run --check`.

The start message arrives once the run has produced a W&B URL, so the link to
watch comes with it. The finish message carries the exit status, the wall time,
the same link, and the last lines of output — usually where the reason for a
failure is. Mail is best-effort: if the server is unreachable the run continues
and says so on stderr, because losing a notification must never cost a training
run.

Python should be unbuffered (`-u`), or the W&B link may not be captured before
the start notification is sent.

For a multi-stage chain, use `scripts/run_all_stages.sh`. It wraps **each stage
separately**, so every stage reports its own start and finish with its own W&B
link. It runs a preflight before anything launches, stops at the first failure,
and sanity-checks every checkpoint it hands to the next stage. `--smoke`
rehearses the identical code path in minutes; `--preflight` runs the checks and
nothing else.

```bash
screen -dmS chain ./scripts/run_all_stages.sh
```

## Fine-tuning an existing chain

`run_all_stages.sh --finetune` warm-starts SR1 and SR2 from an existing
checkpoint bundle (`$SPECSR_INIT_CK`) through
`specsr train sr1|sr2 --init-ckpt`, with the shortened, lower-LR schedules in
`configs/finetune/*.yaml`. Two rules keep this safe:

- The architecture keys in `configs/finetune/*.yaml` mirror the init
  checkpoints and **must not drift from them**; `--init-ckpt` loads strictly, so
  a mismatch fails at launch rather than training an inconsistent model.
- The redshift head never warm-starts. It is cheap to train, and its current
  architecture is incompatible with checkpoints from earlier versions.

The typical use is a chain-wide refresh after the redshift head changes: SR2 is
conditioned on the head's estimate, so a new head means SR2 must adapt.
`scripts/finetune_and_zarms.sh` runs the whole sequence — fine-tune chain,
prediction-cache rebuild, then the redshift-comparison arms — as one launch:

```bash
screen -dmS finetune ./scripts/finetune_and_zarms.sh
```

## What every run records

**Metrics and figures.** Beyond scalar losses, each stage logs a validation
**figure** to Weights & Biases every `plot_every` epochs (default 5), plus a
final panel at the end of training, under a `val/0` key so it sorts above the
scalar charts:

| stage | panel | key |
|---|---|---|
| SR1, SR2 | LR input / SR prediction / HR target, one validation spectrum | `val/0_spectra_LR_SR_HR` |
| ZHead | predicted vs true redshift, with MAE, RMSE, NMAD, median \|Δz\|/(1+z), outlier rate | `val/0_z_pred_vs_true` |

This is not decoration. Both of this project's expensive failures were invisible
in scalars: SR2 can manufacture emission lines in the wrong place while its loss
curve looks healthy, and a redshift run's scalars cannot distinguish ordinary
scatter about the diagonal from a second locus off it caused by a misidentified
line. The redshift panel is drawn by
{func}`specsr.plotting.plot_redshift_panel`, the same function behind the
published figure, so the outlier rate watched during training and the one printed
in the paper are the same statistic.

Helpers live in {mod}`specsr.wandb_plots` and swallow their own exceptions: a
plotting bug logs a warning and the run continues.

**Weights and provenance.** Every stage writes its checkpoint to `--out-dir` and
uploads it to W&B as an artifact, together with `config_resolved.yaml` and
`run_manifest.json` — the dataset, split file, upstream checkpoints, git commit
and W&B run ID. Metrics alone are not a record: a checkpoint you cannot say how
you produced is not reproducible, and local disk is not a backup.

## Splits

Training scripts obtain their split through
{func}`specsr.data.splits.get_training_split`, which splits over **parent
galaxies** rather than rows and refuses to return a leaking split. All three
stages and every evaluation script must use the same split for results to be
comparable; this is guaranteed by keying the split cache on the dataset hash
*and the requested fractions*.

The default is **80/20** with no test partition, and the evaluation side holds
**original spectra only** — 572 real galaxies. Both details matter:

- The built product augments *every* training galaxy at 21 rows each, so an
  unfiltered 20% partition would be ~95% synthetic rows. That reports performance
  on perturbations rather than observations, and computes statistics over 21x
  correlated samples — inflating apparent *n* and understating uncertainties.
- With no test partition the reported number is selected on the set it is
  reported from, since checkpoint selection monitors it. Pass `val_frac=0.1,
  allow_empty_test=False` to restore the three-way split with a sealed test set.

Switching between the two costs no retraining: galaxies are permuted once under
a fixed seed and train takes the first `train_frac`, so 80/20 and 80/10/10 share
a byte-identical training set.

```{warning}
Do not reuse split files with a `split_` prefix. Those were produced by an
earlier row-wise splitter and leak augmented siblings across the train/test
boundary. The current splitter writes `groupsplit_`-prefixed caches and ignores
the old ones, and `assert_no_group_leakage` refuses to train on a split that
leaks, so the failure is loud rather than silent.
```

## Hyperparameter sweeps

Sweeps are defined per stage. One command runs preflight, creates the sweep and
starts the agent under `screen`:

```bash
./scripts/launch_sweep.sh sr1 --preflight   # checks only
./scripts/launch_sweep.sh sr1               # create and launch
```

It refuses to start when another process holds the GPU — worth keeping on a
shared machine with no reservation system.

All trials within a sweep use identical data splits, preprocessing and frozen
upstream checkpoints, so that only the searched parameters vary. Each trial
writes to its own directory keyed on the W&B run ID, so trials cannot overwrite
one another.

**The sweeps use random search rather than Bayesian optimisation, on purpose.**
Their deliverable is a parameter-importance plot, which W&B derives by fitting a
random forest over the trials; Bayesian search concentrates its samples near the
optimum, leaving the axes with little spread and strong mutual correlation,
which is the regime where forest importances stop being trustworthy. `epochs` is
pinned rather than searched for the same reason — longer runs reach a lower loss
almost by construction and would dominate the ranking.

Two rules the configs encode, both learned the hard way:

- **Every searched name must be one the trainer reads.** An earlier SR2 sweep
  named 55 parameters of which 28 were dead, and eight of its eleven search axes
  were among them — a 36-trial search exploring three real dimensions while
  appearing to explore eleven. `tests/test_sweep_configs.py` now fails a config
  that names anything absent from the stage's `DEFAULTS`.
- **The redshift head is swept once, on `--source sr1`, never per arm.** The
  four arms of the redshift comparison differ only in their input, which is the
  entire point; tuning each separately would make that table a measurement of
  tuning effort instead.
