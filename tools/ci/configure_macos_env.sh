#!/bin/bash
# Homebrew's libomp is built for the runner's macOS, and delocate bundles it into the wheels, so use
# conda-forge's llvm-openmp (built for macOS 11) instead, as scikit-learn does.
set -euxo pipefail

if [[ $(uname -m) == "arm64" ]]; then
    MACOSX_DEPLOYMENT_TARGET=14.0
else
    MACOSX_DEPLOYMENT_TARGET=15.0
fi

PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/openmp"
micromamba create -y -p "$PREFIX" -c conda-forge llvm-openmp=23.1.6

# Meson looks for libomp only under `brew --prefix --installed libomp`, falling back to this path.
OMP_ROOT="$(brew --prefix)/opt/libomp"
brew uninstall --ignore-dependencies libomp 2>/dev/null || true
rm -rf "$OMP_ROOT"
ln -s "$PREFIX" "$OMP_ROOT"
ls -l "$OMP_ROOT/include/omp.h" "$OMP_ROOT/lib/libomp.dylib"

# conda-forge's libomp has the install name @rpath/libomp.dylib.
echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET" >> "$GITHUB_ENV"
echo "LDFLAGS=${LDFLAGS:-} -Wl,-rpath,$OMP_ROOT/lib" >> "$GITHUB_ENV"
