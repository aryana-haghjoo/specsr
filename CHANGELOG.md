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

## [0.1.0] - 2026-08-14

### Added

- Initial public release of `specsr`: a physics-informed, three-stage model that
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
