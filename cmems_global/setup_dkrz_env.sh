#!/usr/bin/env bash
# Set up the pixi-managed Python env for this repo on DKRZ Levante and
# register it as a JupyterHub kernel.
#
# Design notes (sources in the references section below):
#   - Env lives on $HOME (VAST) via pixi's `detached-environments` setting,
#     so the many small files land on the low-latency filesystem DKRZ
#     recommends for conda-style envs. DKRZ explicitly recommends $HOME
#     (VAST) for conda envs and discourages /work (Lustre). [1][2]
#   - The package cache is created under /scratch/<letter>/$USER for the
#     duration of `pixi install` and wiped on exit. No persistent cache,
#     no atime/mtime-purge concerns, no $HOME quota consumed by cached
#     tarballs. /scratch has a 14-day atime-based purge anyway, and
#     pixi/rattler extracts packages preserving build-time mtimes (same
#     gotcha as classic conda), so leaning on persistent /scratch cache
#     is fragile. Re-download cost on DFN is negligible. [3]
#   - The env is self-contained after `pixi install` completes: cache
#     files are hardlinks (same-fs) or copies (cross-fs); deleting the
#     cache only drops one link and the env keeps the inode. Verified
#     empirically in docker.
#   - The Jupyter kernel.json invokes `pixi run --manifest-path` so that
#     pixi's activation (PATH, CONDA_PREFIX, [activation.env]) is applied
#     properly — not just a bare python binary. Modeled on the conda-run
#     pattern in the geomar parcels DKRZ setup. [4][5]
#   - The kernel.json sets `env.PATH` explicitly because the DKRZ
#     JupyterHub spawner does not source ~/.bashrc, so $HOME/.pixi/bin
#     must be put on PATH inside the kernel spec itself. [5]
#
# References:
#   [1] DKRZ — Python on Levante:
#       https://docs.dkrz.de/doc/levante/code-development/python.html
#   [2] DKRZ — File Systems:
#       https://docs.dkrz.de/doc/levante/file-systems.html
#   [3] DKRZ — Singularity (analogous /scratch cache pattern):
#       https://docs.dkrz.de/doc/levante/containers/singularity.html
#   [4] geomar-od-lagrange/2025_dkrz_setup (reference parcels setup):
#       https://github.com/geomar-od-lagrange/2025_dkrz_setup
#   [5] DKRZ — JupyterHub kernels:
#       https://docs.dkrz.de/doc/software&services/jupyterhub/kernels.html
#   Background on pixi knobs used here:
#   [6] pixi — detached-environments / cache-dir config:
#       https://pixi.sh/latest/reference/pixi_configuration/
#
# Usage:
#   bash dkrz/setup_dkrz_env.sh              # install + register kernel
#   bash dkrz/setup_dkrz_env.sh --install-only
#   bash dkrz/setup_dkrz_env.sh --register-kernel-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath $REPO_ROOT)"
KERNEL_NAME="$(basename "$REPO_ROOT")"
KERNEL_DISPLAY="Pixi: $KERNEL_NAME"
PIXI_BIN="$HOME/.pixi/bin/pixi"

mode="all"
case "${1:-}" in
  --install-only)        mode="install" ;;
  --register-kernel-only) mode="kernel"  ;;
  "" )                   mode="all"     ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- pixi present?
if [[ ! -x "$PIXI_BIN" ]]; then
  cat >&2 <<EOF
pixi not found at $PIXI_BIN.
Install it once with:

  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="\$HOME/.pixi/bin:\$PATH"   # add to ~/.bashrc

then re-run this script.
EOF
  exit 1
fi

# ------------------------------------------------------- global pixi config
# Park per-project envs on $HOME (VAST), not next to pixi.toml on /work.
"$PIXI_BIN" config set --global detached-environments "$HOME/pixi_envs"

# Default cache location for any bare `pixi add` / `pixi install` call:
# /scratch, so $HOME never accumulates cached tarballs. The install wrapper
# below still overrides this with an ephemeral mktemp dir via PIXI_CACHE_DIR
# (env var beats config), but bare calls outside the wrapper now fail safe
# to /scratch where DKRZ's purge cleans up after them.
SCRATCH_BASE="/scratch/${USER:0:1}/$USER"
if [[ -d "$SCRATCH_BASE" ]]; then
  mkdir -p "$SCRATCH_BASE/pixi-cache-default"
  "$PIXI_BIN" config set --global cache.root "$SCRATCH_BASE/pixi-cache-default"
fi

if [[ "$mode" == "all" || "$mode" == "install" ]]; then
  if [[ ! -d "$SCRATCH_BASE" ]]; then
    echo "expected scratch dir $SCRATCH_BASE does not exist — are you on Levante?" >&2
    exit 1
  fi

  # ---------------------------------------------- ephemeral cache, wipe on exit
  CACHE_DIR="$(mktemp -d "$SCRATCH_BASE/pixi-cache.XXXXXX")"
  trap 'rm -rf "$CACHE_DIR"' EXIT
  echo "using ephemeral cache: $CACHE_DIR"

  PIXI_CACHE_DIR="$CACHE_DIR" "$PIXI_BIN" install --manifest-path "$REPO_ROOT/pixi.toml"
fi

if [[ "$mode" == "all" || "$mode" == "kernel" ]]; then
  # ------------------------------------------------ register JupyterHub kernel
  KDIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
  mkdir -p "$KDIR"
  cat > "$KDIR/kernel.json" <<EOF
{
  "argv": [
    "$PIXI_BIN",
    "run",
    "--manifest-path", "$REPO_ROOT/pixi.toml",
    "python", "-m", "ipykernel_launcher",
    "-f", "{connection_file}"
  ],
  "display_name": "$KERNEL_DISPLAY",
  "language": "python",
  "env": {
    "PATH": "$HOME/.pixi/bin:/usr/bin:/bin"
  }
}
EOF
  echo "registered kernel: $KDIR/kernel.json"
fi

echo "done."
