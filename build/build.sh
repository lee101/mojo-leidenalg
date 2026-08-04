#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/dist"
mojo build --emit shared-lib "$root/src/leiden.mojo" -o "$root/dist/libmojo-leidenalg.so"
