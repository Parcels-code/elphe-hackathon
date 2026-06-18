#!/usr/bin/env bash
# Configure pixi's global knobs for DKRZ Levante. Run this once (and after a
# fresh pixi install); it is global and idempotent, safe to re-run.
#   - per-project envs on $HOME (VAST), not on /work (Lustre) next to pixi.toml
#   - default package cache on /scratch, so $HOME never fills with tarballs
# Rationale and doc links: see README.md ("Background / references").
set -euo pipefail

PIXI_BIN="$HOME/.pixi/bin/pixi"
[[ -x "$PIXI_BIN" ]] || { echo "pixi not found at $PIXI_BIN — install with: curl -fsSL https://pixi.sh/install.sh | bash" >&2; exit 1; }

# Park per-project envs on $HOME (VAST), not next to pixi.toml on /work.
"$PIXI_BIN" config set --global detached-environments "$HOME/pixi_envs"

# Default cache on /scratch so `pixi install` writes tarballs there (DKRZ
# purges it; $HOME quota stays clean) while the envs live under detached $HOME.
SCRATCH_BASE="/scratch/${USER:0:1}/$USER"
if [[ -d "$SCRATCH_BASE" ]]; then
  mkdir -p "$SCRATCH_BASE/pixi-cache-default"
  "$PIXI_BIN" config set --global cache.root "$SCRATCH_BASE/pixi-cache-default"
else
  echo "note: $SCRATCH_BASE not found — leaving cache.root at the pixi default" >&2
fi
echo "pixi configured."
