# Data

Raw JWST/JADES data is **not** distributed with the package and is not stored in
the repository. Only derived products are built here, and all of them are
regenerable.

```bash
export SPECSR_JADES_ROOT=/path/to/JADES_data   # contains DR3/, DR4/, ...
export SPECSR_DATA_DIR=/path/to/derived
specsr build-dataset --release DR4
```

You only need the raw data to **rebuild the dataset or retrain**. To run the
released models on your own spectra, or to work through the tutorials, you do
not: weights download from the Hugging Face Hub automatically and the tutorials
carry their own small samples.

## The layout `specsr` expects

```
$SPECSR_JADES_ROOT/
└── DR4/
    ├── Combined_DR4_external_v1.2.1.fits          # z_Spec, quality flags, MUV
    ├── goods-n/
    │   └── spectra/
    │       └── <filter>-<grating>/                # clear-prism, f290lp-g395m, ...
    │           └── <tier>/                        # goods-n-mediumjwst, ...
    │               └── hlsp_jades_jwst_nirspec_<tier>-<id>_<filter>-<grating>_v1.0_x1d.fits
    └── goods-s/
        └── spectra/ ...
```

`Combined_DR4_external_v1.2.1.fits` is required as well as the spectra: it
supplies `z_Spec` and its quality flags, and only galaxies with a secure
redshift (flags A/B/C) are kept.

Only the **1D extractions** (`x1d.fits`) are read. The 2D `s2d.fits` products,
NIRCam cutouts and the imaging releases are never opened, and they are the bulk
of the archive — skipping them saves hours and a great deal of disk.

## Downloading DR4

DR4 is served from the JADES collaboration's own site as a browsable tree whose
layout is already the one above, so `wget` can mirror exactly the four
dispersers the package reads:

- **Release page:** <https://jades-survey.github.io/scientists/data.html>
- **Data tree:** <https://jades.herts.ac.uk/DR4/>

```bash
export SPECSR_JADES_ROOT=$HOME/JADES_data
mkdir -p "$SPECSR_JADES_ROOT/DR4"

# The redshift and quality-flag catalogue (~90 MB).
wget --directory-prefix "$SPECSR_JADES_ROOT/DR4" \
     https://jades.herts.ac.uk/DR4/Combined_DR4_external_v1.2.1.fits

# The 1D spectra: prism plus the three medium gratings, both fields.
for arm in clear-prism f070lp-g140m f170lp-g235m f290lp-g395m; do
  for field in goods-n goods-s; do
    wget --recursive --no-parent --no-host-directories --continue \
         --accept '*x1d.fits' \
         --directory-prefix "$SPECSR_JADES_ROOT" \
         "https://jades.herts.ac.uk/DR4/$field/spectra/$arm/"
  done
done
```

`--accept '*x1d.fits'` is what restricts this to the 1D products; `--continue`
makes it resumable, which matters because this is tens of thousands of small
files. The `f290lp-g395h` directory is deliberately absent from the loop — the
high-resolution grating is not part of the paired sample.

```{note}
DR4 is not yet on MAST. The MAST High Level Science Product page
(<https://archive.stsci.edu/hlsp/jades>, DOI
[10.17909/8tdj-8n28](https://doi.org/10.17909/8tdj-8n28)) serves DR1-DR3, and
the JADES team states that DR4 will follow. Until it does, the collaboration
site above is the download path.
```

### Earlier releases

DR3 and earlier are on MAST and mirror the same way:

```bash
wget --recursive --no-parent --no-host-directories --cut-dirs=2 --continue \
     --accept '*x1d.fits' \
     --directory-prefix "$SPECSR_JADES_ROOT" \
     https://archive.stsci.edu/hlsps/jades/dr3/goods-s/
```

## What the build does

1. **Ingest** — read NIRSpec `x1d` products for the prism and the medium
   gratings (G140M, G235M, G395M).
2. **Pair** — match prism and grating observations of the same target, and
   stitch the three gratings into a single reference covering 1-5 µm.
3. **Quality filter** — apply coverage and signal-to-noise cuts.
4. **Resample** — put both members of each pair on the common logarithmic grid,
   by integration rather than interpolation, so line flux is preserved.
5. **Augment** — generate stochastic realizations per galaxy (Gaussian redshift
   offsets and flux perturbations), recording each row's parent galaxy.

## Provenance and the parent galaxy

Augmentation multiplies each galaxy into many highly correlated rows. Products
therefore carry an explicit parent identifier, and `(ra, dec, field)` is
preserved unchanged so that provenance can always be recovered even from older
products that predate the explicit column.

This is what makes leak-free splitting possible — see
{mod}`specsr.data.splits` and the [training guide](training.md).

```{important}
Never split these datasets by row. All realizations of a galaxy must fall on the
same side of a train/test boundary, or held-out metrics will be optimistic.

The builder goes further: it decides the split *before* augmenting and expands
only the training galaxies, so a held-out galaxy contributes exactly one row —
its real spectrum — and nothing synthetic derived from it exists anywhere in the
product. `augment_train_only`, `split_train_frac` and `split_seed` are recorded
in the `.npz` so a consumer can check that the split it draws is the one the file
was built for.
```

## Releases

| Release | Contents | Use |
|---|---|---|
| DR3 | Prism + medium gratings | Superseded |
| DR4 | Prism + medium gratings, GOODS-N/S | Current training set |
| DR5 | NIRCam imaging and photometric catalogues only | Sample characterisation |

DR5 contains no spectroscopy; its photometric catalogues are used to
characterise the spectroscopic sample.
