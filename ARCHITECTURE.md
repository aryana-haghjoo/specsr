<!--
This is the PUBLIC architecture document, shipped to the released repository as
ARCHITECTURE.md by scripts/make_public_release.sh.

It is a different file from the development ARCHITECTURE.md in the private
repository. That one is a decisions log -- why code moved between directories,
which experiment forced which change, what a given failure cost -- and is
addressed to whoever maintains the working tree. This one describes the design
as it stands, for somebody reading or extending the code.
-->

# Architecture

How `specsr` is put together, and the constraints the code enforces on itself.
For how to *use* it, start with the [documentation](https://aryana-haghjoo.github.io/specsr/)
or the notebooks in `tutorials_for_user/`.

## The model

![Three-stage spectral super-resolution pipeline](docs/_static/architecture.png)

Three stages, trained in order. Each consumes the frozen output of the one
before it, so an error in an early stage is inherited rather than corrected.

**SR1 — super-resolution backbone.** A 1D residual CNN (16 blocks x 120
channels) that maps a prism spectrum on the common grid to a grating-like
reconstruction on the same grid, emitting a mean and a per-pixel log-variance.
It is deliberately conservative: it sharpens structure the input supports and is
not asked to invent line profiles.

**ZHead — redshift inference.** Six dilated convolution blocks x 128 channels
with attention pooling over wavelength, reading SR1's mean and log-variance. The
output is a softmax distribution over 1,024 redshift bins spanning 0 <= z <= 15;
the point estimate is a soft-argmax in a window around the modal bin, and the
spread comes from the full distribution.

Two properties matter and both are structural rather than tuned. The head must
be **position-aware** — on a logarithmic wavelength grid a redshift is a
translation, so a convolutional trunk followed by a global mean pool cannot
measure one even in principle. And it must predict a **distribution** rather than
a point: the characteristic failure is a misidentified line, which puts a second
mode elsewhere, and a Gaussian head answers such a posterior with the mean,
landing in the empty valley between the modes.
`tests/test_models.py` pins both properties against synthetic data.

**SR2 — physics-informed residual refiner.** Multi-head self-attention over 98
emission-line tokens (`specsr.models.lines`), each emitting a parametric
Gaussian — amplitude, width, offset — gated by a presence probability, plus a
CNN branch for smooth continuum corrections. The two branches sum to a residual
that is added to SR1's output.

SR2's five input channels are the prism spectrum, SR1's mean and sigma, a line
mask `m(lambda; z, sigma_z)` built at the *predicted* redshift, and the redshift
itself. There is no teacher forcing anywhere: SR2 only ever sees redshifts ZHead
inferred, never the catalogue value, so the chain behaves at training time the
way it behaves at inference time.

The presence gate is **supervised**, not merely penalised — trained by
class-balanced cross-entropy against the lines the reference spectrum actually
shows. A blanket sparsity penalty applies the same downward gradient to every
line logit whether or not the line is real; because the model forms
`amplitude * presence`, that becomes a constant multiplier on every amplitude and
can silently disable the entire line branch while every loss curve still looks
healthy. `train/presence_sep` — the ratio of gate output on real lines to absent
ones — is logged and guarded for exactly this reason.

Line-mask widths grow with ZHead's predicted `sigma_z`, capped: a spectrum whose
redshift is uncertain gets a wider mask rather than a confidently misplaced
narrow one. The cap matters because the same mask weights SR2's in-line loss
term, and an uncapped mask on a hopeless redshift would relax that term into a
plain reconstruction loss without any visible symptom.

## The wavelength grid

Everything runs on one grid: **logarithmic, constant R = 4,000, 1.0-5.3 µm,
6,671 pixels** (`specsr.data.grid.DEFAULT_GRID`). A log grid holds
lambda/dlambda fixed, which is how a spectrograph actually samples — a fixed
*fractional* step matches the instrument at every wavelength instead of being too
coarse in the blue and wasteful in the red.

Two rules follow, and both are load-bearing:

- **Regrid the low-resolution data up onto the high-resolution sampling; never
  resample the high-resolution spectra down.** A linear grid coarse enough to
  undersample the gratings makes the *grid*, not the model, the ceiling on
  achievable sharpness — and the resulting model looks like it has hit a
  fundamental limit.
- **Resampling integrates; it does not interpolate.** The physically meaningful
  quantity in an emission line is its integrated flux, an area, not the height of
  the curve at particular abscissae. `resample_flux_conserving` is the default
  for that reason, and `SpecSRPipeline` applies it to any input handed to it on a
  different grid.

## Package layout

```
src/specsr/
  data/          ingest, grid, stitching, augmentation, datasets, group splits
  models/        SR1, ZHead, SR2, the line catalogue, checkpoint loaders
  training/      the three training loops, losses, redshift transform
  inference/     SpecSRPipeline
  evaluation/    split selection, chain inference, the published figures
  plotting.py    every figure in the paper
  metrics.py     linefit.py     line fitting and summary statistics
  checkpoints.py on-demand weight fetching from the Hugging Face Hub
  paths.py       environment-driven data, cache and output directories
configs/         stage configs and W&B sweep definitions
scripts/         thin entry points: argument parsing and file IO only
tests/           unit tests plus checkpoint-reproduction guards
```

**All library code lives under `src/specsr/`.** `scripts/` parses arguments and
moves files; it defines no model, dataset or loss. This is not style — the
package was once a parallel implementation that nothing imported during training,
and an unexercised copy drifts silently. It did, within a day, and anyone loading
a checkpoint through the package would have got a different network from the one
that trained. `tests/test_package_reproduces_checkpoints.py` loads real
checkpoints through the *package* classes and asserts on measured numbers, so the
drift cannot recur unnoticed.

## Data flow

```
raw JADES x1d products
   |  specsr.data.ingest      read prism and G140M/G235M/G395M extractions
   |  specsr.data.stitch      one 1-5 µm grating reference per galaxy
   |  quality cuts            coverage and S/N; secure catalogue redshifts only
   |  specsr.data.grid        both members resampled onto DEFAULT_GRID
   |  specsr.data.augment     training galaxies only -> 21 rows each
   v
paired .npz  (LR, HR, sigma, z, parent id, provenance)
   |  specsr.data.splits      split over parent galaxies
   v
specsr train sr1 -> zhead -> sr2        each freezing the previous stage
   v
specsr infer / evaluate,  SpecSRPipeline.from_pretrained()
```

## Invariants the code enforces

These are the assumptions that, when broken, produce results that look correct.
Each is checked in code rather than left to discipline.

**Splits are drawn over parent galaxies, never over rows.** Augmentation
multiplies each galaxy into 21 highly correlated rows differing only by a small
redshift offset and noise. A row-wise split puts near-duplicates of every
held-out galaxy into training and every held-out number comes out optimistic.
`specsr.data.splits` groups by parent and refuses to return a leaking split;
`assert_no_group_leakage` is called before training, not after.

**Only training galaxies are augmented.** The builder decides the split *before*
augmenting, so a held-out galaxy contributes exactly one row — its real observed
spectrum — and `train + val` accounts for every row in the file. A row that must
never be read is a trap for whoever does not know that. The builder then
re-derives the split from the file it just wrote and refuses to finish if the two
disagree.

**Evaluation is on original spectra only.** The built product augments every
galaxy, so an unfiltered held-out partition would be ~95% synthetic rows: it
would report performance on perturbations rather than observations, over
correlated samples, inflating apparent *n* and understating uncertainties. The
default configuration is 80/20 over galaxies with **no sealed test partition**,
which means the reported number is selected on the set it is reported from; pass
`val_frac=0.1, allow_empty_test=False` to `get_training_split` to restore a
three-way split with a withheld test set. Switching costs no retraining, because
galaxies are permuted once under a fixed seed and training takes the first
`train_frac`.

**Upstream checkpoints are named explicitly, never discovered.** Every stage
takes its predecessors' paths on the command line, so a checkpoint records what
it was trained on top of. Evaluating against a directory a running job is
overwriting compares refiners sitting on top of *different* backbones; both
numbers are real and neither is comparable to the other.

**Checkpoints are never written into the source tree.** Every stage takes
`--out-dir`, which makes it structurally impossible for a rehearsal run to
address a real run's output path.

**A checkpoint carries what is needed to decode it.** The redshift head stores
its normalisation bounds; without them the pipeline cannot convert a network
output back into a redshift, and it raises rather than guessing. Guessing does
not fail visibly — it returns confident, plausible, wrong redshifts.

**Verify checkpoints by measurement, not by hash.** A hash proves two files
agree; it does not prove either is correct. Check a checkpoint's stored config,
or measure it against a known number.

**Every run logs figures, not only scalars.** Each stage writes a validation
panel to Weights & Biases alongside its metrics, and the redshift panel is drawn
by the same `plotting.plot_redshift_panel` the published figure uses. Both of
this project's expensive failures were invisible in loss curves: a refiner can
manufacture lines in the wrong place while its loss falls, and scalar redshift
metrics cannot distinguish ordinary scatter about the diagonal from a second
locus off it.

## Numerical stability

Gradient clipping is not a defence against an exploding gradient.
`torch.nn.utils.clip_grad_norm_` scales by `max_norm / total_norm`; a single
non-finite element makes the norm infinite, the scale zero, and their product
NaN — so clipping *launders* an overflow into every weight, and an adaptive
optimiser's moments then make it permanent. All three stages call
`specsr.training.common.clip_grad_or_skip`, which skips the optimiser step when
the gradient norm is not finite. SR2's line branch is the one that needs it: it
runs at a gradient norm around 0.04 and spikes many orders of magnitude above
that, and a meaningful fraction of its steps are skipped. This is containment,
not a fix.

## Extending it

- **A new spectral feature** goes in `specsr.models.lines`. The list order *is*
  the token order the checkpoints were built with, so append rather than insert,
  and expect existing SR2 weights not to load if you change the count.
- **A different wavelength grid** means rebuilding the dataset and retraining
  from SR1 up: the grid is baked into every checkpoint's input length.
- **A new stage or loss term** goes under `src/specsr/training/`, with its
  hyperparameters added to that stage's `DEFAULTS`. Sweep configs are validated
  against `DEFAULTS`, so a search axis the trainer does not read fails the test
  suite rather than quietly wasting a sweep.
- **A new figure** goes in `specsr.plotting` as a function, called from a thin
  script. Wrap it in `plt.rc_context(PAPER_RC)` rather than setting fonts
  locally.
