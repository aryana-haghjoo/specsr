# Installation

```bash
pip install "specsr[hub]"
```

Extras select what comes with it: `hub` adds weight download from the Hugging
Face Hub, `train` adds the training stack (wandb, pandas), `zcompare` adds the
alternative redshift estimators used for the independent cross-check, and `all`
is everything.

## PyTorch and CUDA

`specsr` depends on PyTorch but deliberately does not pin a CUDA build. On a GPU
machine, install the torch build matching your driver **first**, from the
[official index](https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "specsr[hub]"
```

To track `main` rather than a release:

```bash
pip install "git+https://github.com/aryana-haghjoo/specsr.git#egg=specsr[hub]"
```

```{warning}
Installing `specsr` into an environment with a pre-existing CUDA-enabled torch
can cause pip to resolve a different set of `nvidia-*` runtime wheels, producing
errors such as `undefined symbol: ncclCommWindowRegister` at import time. If that
happens, reinstall the `nvidia-*` pins listed in your torch distribution's
metadata:

    grep -E "^Requires-Dist: nvidia" \
      $(python -c "import torch,pathlib;print(pathlib.Path(torch.__file__).parent.parent)")/torch-*.dist-info/METADATA
```

## Development install

```bash
git clone https://github.com/aryana-haghjoo/specsr
cd specsr
pip install -e ".[dev,train,hub]"
pytest
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `SPECSR_JADES_ROOT` | Root of the raw JADES tree (`DR3/`, `DR4/`, ...) | unset — required to build datasets |
| `SPECSR_DATA_DIR` | Where derived products are written | `./data` |
| `SPECSR_OUTPUT_DIR` | Where figures, predictions and evaluation tables are written | `./outputs` |
| `SPECSR_CACHE_DIR` | Downloaded weights and caches | `~/.cache/specsr` |
| `SPECSR_CHECKPOINT_DIR` | Load weights from a local directory instead of the Hub | unset |
| `SPECSR_CHECKPOINT_REPO` | Override the Hub model repo | `aryana-haghjoo/specsr` |
| `SPECSR_CHECKPOINT_REVISION` | Pin a Hub branch, tag or commit | `main` |

## Optional: run notifications

Training runs can email you when they start and finish. It is opt-in and uses
**your own address and mail server** -- nothing is configured by default and
nothing is sent until you ask for it:

```bash
./scripts/notify-run --setup
```

That asks for the address to notify, the SMTP server to send through, and the
credentials if the server needs them, then writes `~/.specsr_notify.conf` at
mode 600 and offers to send a test message. Use an app-specific password rather
than your account password: it is scoped to sending mail and can be revoked on
its own.

To do it by hand, or in CI and containers where a file on disk is the awkward
part, every setting is equally an environment variable:

```bash
SPECSR_NOTIFY_TO=you@example.edu
SPECSR_NOTIFY_SMTP_HOST=smtp.example.edu
SPECSR_NOTIFY_SMTP_PORT=587
SPECSR_NOTIFY_SMTP_USER=you@example.edu
SPECSR_NOTIFY_SMTP_PASS=your-app-password
```

`./scripts/notify-run --check` reports what is configured, without printing the
password. See the [training guide](training.md) for what the messages contain.
