#!/usr/bin/env bash
# Register one JupyterHub kernel per named pixi environment in pixi.toml.
# Each kernel launches `pixi run --environment <env>` so pixi activation (PATH,
# CONDA_PREFIX, [activation.env]) applies — not a bare python. PATH is pinned in
# the kernel spec because the DKRZ spawner does not source ~/.bashrc. See README.
set -euo pipefail

# Workspace root (where pixi.toml lives) is the parent of this scripts/ dir;
# resolved from the script location, so it works regardless of $PWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_NAME="$(basename "$REPO_ROOT")"
PIXI_BIN="$HOME/.pixi/bin/pixi"
[[ -x "$PIXI_BIN" ]] || { echo "pixi not found at $PIXI_BIN — install with: curl -fsSL https://pixi.sh/install.sh | bash" >&2; exit 1; }

# Named environments = every environment in pixi.toml except bare `default`.
mapfile -t ENVS < <(
  "$PIXI_BIN" workspace environment list --manifest-path "$REPO_ROOT/pixi.toml" \
    | sed -n 's/^- \([^:]*\):.*/\1/p' | grep -vx default
)
[[ "${#ENVS[@]}" -gt 0 ]] || { echo "no named environments in pixi.toml" >&2; exit 1; }

for env in "${ENVS[@]}"; do
  KDIR="$HOME/.local/share/jupyter/kernels/${KERNEL_NAME}-${env}"
  mkdir -p "$KDIR"
  cat > "$KDIR/kernel.json" <<EOF
{
  "argv": [
    "$PIXI_BIN",
    "run",
    "--manifest-path", "$REPO_ROOT/pixi.toml",
    "--environment", "$env",
    "python", "-m", "ipykernel_launcher",
    "-f", "{connection_file}"
  ],
  "display_name": "Pixi: $KERNEL_NAME ($env)",
  "language": "python",
  "env": {
    "PATH": "$HOME/.pixi/bin:/usr/bin:/bin"
  }
}
EOF
  echo "registered kernel: $KDIR/kernel.json"
done
