#!/usr/bin/env bash
# Create the "gp_env" conda environment inside the group folder instead of the home folder.
# Safe to re-run: an existing environment is left untouched.
#
# The home quota on the cluster is small and a conda environment is not. This script points
# both the environment folder and the package cache at the shared miniconda installation, so
# that plain `conda env create -f gp_env.yml` and `conda activate gp_env` keep working and
# land in the group folder from now on. Keeping the package cache there too matters: conda
# can only hardlink a cached package into an environment on the same filesystem, otherwise it
# copies every file.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash setup_gp_env.sh [options]

  --root PATH   shared miniconda installation
                (default: /groups/ReismanLab/miniconda3, or the /resnick path if that is the one)
  --all-users   write the configuration into the shared installation ($ROOT/.condarc) so that
                every group member inherits it, instead of into your own ~/.condarc
  --no-test     skip the test suite at the end
USAGE
}

die() { printf 'setup_gp_env: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${CONDA_SHARED_ROOT:-}
ALL_USERS=0
RUN_TESTS=1

while (($#)); do
    case $1 in
        --root)      ROOT=${2:?--root needs a path}; shift 2 ;;
        --all-users) ALL_USERS=1; shift ;;
        --no-test)   RUN_TESTS=0; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           usage >&2; die "unknown argument: $1" ;;
    esac
done

if [[ -z $ROOT ]]; then
    for group in /groups/ReismanLab /resnick/groups/ReismanLab; do
        if [[ -d $group ]]; then
            ROOT=$group/miniconda3
            break
        fi
    done
    [[ -n $ROOT ]] || die "no ReismanLab group folder found; pass --root /path/to/shared/miniconda3"
fi

ENVS_DIR=$ROOT/envs
PKGS_DIR=$ROOT/pkgs
PREFIX=$ENVS_DIR/gp_env

command -v conda >/dev/null 2>&1 || die "conda is not on PATH; try: source $ROOT/etc/profile.d/conda.sh"
[[ -f $REPO/gp_env.yml ]] || die "gp_env.yml not found next to this script ($REPO)"

step "shared folders under $ROOT"
mkdir -p "$ENVS_DIR" "$PKGS_DIR" || die "cannot create $ENVS_DIR and $PKGS_DIR"
[[ -w $ENVS_DIR && -w $PKGS_DIR ]] || die "no write permission on $ENVS_DIR or $PKGS_DIR"
# setgid, so that whatever lands there stays readable and writable for the group
chmod g+ws "$ENVS_DIR" "$PKGS_DIR" 2>/dev/null || true
printf 'environments: %s\npackage cache: %s\n' "$ENVS_DIR" "$PKGS_DIR"

if ((ALL_USERS)); then
    CONDARC=$ROOT/.condarc
else
    CONDARC=$HOME/.condarc
fi
step "conda configuration in $CONDARC"
# Touch first: in a file it creates itself, conda also writes a "channels: defaults" block.
[[ -e $CONDARC ]] || touch "$CONDARC" || die "cannot write $CONDARC"
conda config --file "$CONDARC" --prepend envs_dirs "$ENVS_DIR"
conda config --file "$CONDARC" --prepend pkgs_dirs "$PKGS_DIR"
# "already in the list, moving to the top" on a second run is the expected no-op.

# An empty DescriPyTor/ means the submodule was never checked out; the extraction step needs it.
if [[ -z $(ls -A "$REPO/DescriPyTor" 2>/dev/null) ]] && git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    step "DescriPyTor submodule"
    git -C "$REPO" submodule update --init
fi

step "environment gp_env"
if [[ -f $PREFIX/conda-meta/history ]]; then
    echo "already exists at $PREFIX, left untouched"
    echo "(to refresh it: conda env update --file $REPO/gp_env.yml --prefix $PREFIX --prune)"
elif [[ -d $PREFIX ]]; then
    die "$PREFIX exists but is not a conda environment (an interrupted run?); remove it and start again"
elif ! conda env create --file "$REPO/gp_env.yml" --prefix "$PREFIX"; then
    # Recent conda versions refuse to touch repo.anaconda.com before its Terms of Service have
    # been accepted once per account. Accepting them is your decision, not this script's.
    cat >&2 <<'EOF'

If the error above is about Terms of Service, run the two "conda tos accept" commands conda
printed, then run this script again; it continues where it stopped.
EOF
    exit 1
fi

if ((RUN_TESTS)); then
    step "test suite"
    (cd "$REPO" && conda run --prefix "$PREFIX" --no-capture-output pytest tests -m "not slow")
fi

cat <<EOF

Done. gp_env lives in $PREFIX; the only file changed outside it is $CONDARC.
Activate it from anywhere, by name:

    conda activate gp_env
EOF
