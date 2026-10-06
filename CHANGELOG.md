# Changelog

All notable changes to `specsr` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
This is the PUBLIC changelog, shipped to the released repository as
CHANGELOG.md by scripts/make_public_release.sh.

It is deliberately not the same file as the development CHANGELOG.md in the
private repository. That one records the internal history -- which run
directory produced which checkpoint, which review comment prompted which
change -- and is addressed to whoever maintains the working tree. This one is
addressed to somebody installing the package, and starts at the first public
release. Add entries here when a change affects users.
-->

## [Unreleased]

## [1.2.0] - 2026-10-06

### Added

- `specsr.linefit.measure_line_fluxes` and `line_flux_from_fit`: the integrated
  flux of a line from the same Gaussian fit that gives its signal-to-noise,
  `sqrt(2 pi) * A * sigma`.
- `specsr.linefit.fit_hbeta_oiii`: a joint fit of H-beta, [O III] 4959 and
  [O III] 5007 as three Gaussians of a shared width. These lines are one feature
  at prism resolution, and a single Gaussian centred on H-beta widens to cover
  all three and returns up to ten times the true flux.
- `flux_scale` and `flux_unit` arguments to
  `specsr.plotting.plot_line_flux_comparison`, so its axes can be drawn in
  physical units.

### Changed

- **`specsr evaluate line-flux` now measures line flux from Gaussian fits, not
  from a fixed velocity aperture, and its numbers have changed.** The figure
  used to sum continuum-subtracted flux within +/-500 km/s of each line. That
  aperture holds the whole line at the resolution of the reference but about a
  fifth of it at prism resolution, so the row for the low-resolution input read
  as a loss of 70-90% of the flux when the flux was present and unresolved.
  Every spectrum is now measured through its own profile: a single Gaussian for
  [O II] and H-alpha, the joint fit above for H-beta and [O III] 5007. The
  low-resolution input now sits on the one-to-one line (median ratio to the
  reference 0.68, 0.91, 1.06 and 1.09 for the four lines) and the super-resolved
  output just below it (0.72, 0.87, 0.85, 0.86), where the aperture version gave
  0.10-0.30 and 0.44-0.60. The figure is built from the prediction cache alone;
  `--flux-rows` is still accepted and no longer read.
- The line-flux figure's axes are in erg s^-1 cm^-2. They were unlabelled and in
  the unit of flux density times wavelength, erg s^-1 cm^-2 AA^-1 um, which is
  10^4 times smaller.
- A line enters the line-flux comparison when the reference fit detects it in
  emission at S/N >= 5, the signal-to-noise of the `line-snr` analysis. The
  selection used the aperture flux over its propagated error before.
- README and the evaluation guide state that super-resolution does not improve
  total line fluxes over the low-resolution input.
- The author list of the paper in `CITATION.cff` was updated.

### Internal

- The documentation build no longer fails when an intersphinx inventory host is
  unreachable. It runs with warnings as errors, and intersphinx reports an
  unreachable inventory as an untyped warning that `suppress_warnings` cannot
  target -- so any project the docs cross-reference could veto a deployment.
  docs.scipy.org going down failed the 1.1.0 docs deploy twice with nothing
  wrong in the repository. References into a missing inventory now degrade to
  plain text; every other warning is still an error.

## [1.1.0] - 2026-09-03

### Added

- `specsr archive` and `specsr.checkpoints.archive_dir`, for fetching a
  superseded checkpoint chain from the Hub by name. `specsr archive` lists what
  is available; `specsr archive <name>` prints its directory, downloading it on
  first use. Training and sweep configurations can also name one directly, as
  `--sr1-ckpt archive:release/best_superres_model.pth`. The chains are published
  for provenance and for reproducing earlier numbers; the current models remain
  those served by `get_checkpoint`, and mixing stages between chains is not
  valid.
- `scripts/make_sweep_figure.py` and the `sweeps/` trial tables, which together
  reproduce the hyperparameter-sensitivity figure of the paper's appendix. The
  tables ship with the repository, so the figure regenerates from a clean
  checkout with no Weights & Biases account; `--refresh` re-exports them if you
  have access to the sweeps.

### Fixed

- A comment in `configs/sweeps/sr2.yaml` claimed that freezing the redshift
  head makes the redshift outlier rate constant across trials, so that it drops
  out of the sweep's ranking metric. It does not: the head is applied to SR2's
  output, so SR2's own weights move the prediction even when the head is
  entirely frozen. Freezing it is still correct, because it stops a trial from
  improving its score by degrading the estimator that grades it.
- `specsr evaluate coverage` could place its emission-line markers at a
  redshift belonging to a different galaxy. It looked the redshift up by target
  id alone and took the first matching row of the dataset, but augmented rows
  carry a deliberately shifted redshift and a few target ids occur in both
  survey fields. It now reads original rows only, keyed by field and target id
  together, and reports an error instead of falling back to z = 0.

### Changed

- `specsr evaluate` now accepts `--coverage-target` and `--coverage-field`, so
  the coverage figure can be drawn for a galaxy you name. With no target it
  still falls back to the first one found on disk, which is usually too faint
  for its grating spectra to be legible; goods-s 00197911 (z = 3.062) is a good
  example to pass.
- The line-flux comparison figure now draws the input prism as a second row
  beneath the super-resolved one, sharing axis limits per column, and
  `plot_line_flux_comparison` takes a `products` argument naming the rows. A
  single row states that SR2 recovers about half the reference line flux, which
  reads as a failure until the prism row shows what the same aperture recovers
  from the input. The prism row is not a claim that the prism has lost the
  flux: every panel measures a fixed +/-500 km/s aperture and the prism's line
  is several times wider, so its deficit is flux outside the aperture. Building
  this figure now needs a `flux_conservation` cache carrying the prism column
  and reports what to re-run if given an older one, instead of quietly falling
  back to the single-row version the caption no longer describes.
- `rank_doublet_examples` applies its amplitude-ratio and peak-offset bounds to
  both components of the doublet, not to 5007 alone, and ranks what passes by
  total amplitude error rather than by separation alone. The old bounds let an
  example through with a faithful 5007 beside a 4959 at half the reference
  height or displaced most of the way to its neighbour -- which is what a
  reader looking at the doublet notices first. The defaults tightened with it:
  `max_peak_offset` 0.25 to 0.12, `amp_ratio_range` (0.4, 2.5) to (0.7, 1.3).

### Internal

- The docs workflow no longer runs on a private repository, where GitHub Pages
  is unavailable and every push therefore recorded a failed deployment that was
  never meant to happen. It ships unchanged to the public repository, where it
  runs as before.

## [1.0.2] - 2026-08-15

### Added

- Published to PyPI: `pip install "specsr[hub]"`.
- `scripts/notify-run --setup` asks for your email address and mail server and
  writes the configuration for you, instead of requiring a hand-written file.

### Fixed

- `specsr.__version__` reported a stale hardcoded version rather than the
  installed one.
- `log_checkpoint_artifact` could raise `AttributeError` at the end of training
  when a `wandb/` run directory shadowed an uninstalled `wandb` library.

## [1.0.1] - 2026-08-15

### Added

- First public release of `specsr`: a physics-informed, three-stage model that
  super-resolves JWST/NIRSpec prism spectra (R ~ 100) towards the medium
  gratings (R ~ 1000).
- `SpecSRPipeline.from_pretrained()` downloads the released weights from the
  Hugging Face Hub on first use; no local data or credentials required.
- Command line: `specsr build-dataset | train | infer | evaluate`.
- Three tutorial notebooks with bundled held-out spectra, runnable without the
  survey data.
- `SPECSR_OUTPUT_DIR` collects everything the package writes -- figures,
  predictions, evaluation tables -- in one place, defaulting to `./outputs`.
- `ARCHITECTURE.md` describes the three stages, the wavelength grid, and the
  invariants the code enforces on itself.
- Optional start/finish email notifications for training runs, via
  `scripts/notify-run`. Off unless configured with your own address and mail
  server; unconfigured it is a transparent passthrough.

Archived on Zenodo: [10.5281/zenodo.21943197](https://doi.org/10.5281/zenodo.21943197)
(all versions: [10.5281/zenodo.21943196](https://doi.org/10.5281/zenodo.21943196)).

### Known limitations

These are measured, not suspected, and are documented in the paper:

- **Absolute flux scale is not predicted.** Training standardises each spectrum
  independently, so the model learns a shape-to-shape mapping and returns output
  on the input's scale. Compare in normalised units.
- **Integrated line fluxes are systematically under-recovered**, and recovery
  depends strongly on line brightness. Reconstructions are suited to locating
  and detecting lines rather than to measuring their strengths.
- **The redshift head is a conditioning stage, not a redshift pipeline.** Its
  catastrophic-outlier rate is high enough that a redshift should be checked
  against the line positions it implies.
- **Super-resolution does not improve on direct analysis of the prism** for the
  [O III] doublet ratio test. See the paper for the comparison.
