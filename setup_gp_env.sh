#!/usr/bin/env bash
# Create the "gp_env" conda environment in the group folder, for the whole group to use,
# instead of one private copy per member in the home folder.
# Safe to re-run, by anybody: an existing environment is left untouched.
#
# The home quota on the cluster is small and a conda environment is not. This script points both
# the environment folder and the package cache at the group folder, so that plain
# `conda env create -f gp_env.yml` and `conda activate gp_env` keep working and land there from
# now on. Keeping the package cache next to the environments matters: conda can only hardlink a
# cached package into an environment on the same filesystem, otherwise it copies every file.
#
# The first member to run it builds the environment (~10 min). Everybody else runs the same
# command and it only points their conda at the group folder, a few seconds.

set -euo pipefail

# Never stop on conda's "would you like to send this report to the maintainers? [y/N]" prompt.
export CONDA_REPORT_ERRORS=false
# Whatever we create has to stay readable for the rest of the group; a private umask (0077) would
# produce a shared environment nobody else can use.
umask 022

usage() {
    cat <<'USAGE'
Usage: bash setup_gp_env.sh [options]

  --root PATH   shared folder holding envs/ and pkgs/
                (default: /groups/ReismanLab/miniconda3, or the /resnick path if that is the one)
  --all-users   write the configuration into the shared installation ($ROOT/.condarc) instead of
                your own ~/.condarc; only useful when $ROOT is the conda installation the group
                runs, otherwise no conda ever reads that file
  --no-test     skip the test suite at the end
USAGE
}

die() { printf 'setup_gp_env: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

# `conda env create`, whatever the conda version. Up to conda 22.x, `conda env` is not a subcommand
# but a separate conda-env executable, and that executable is missing from some installations
# (FileNotFoundError: .../bin/conda-env); the same code is then still reachable as a python module.
env_create() {
    local python
    if conda env list >/dev/null 2>&1 </dev/null; then
        conda env create --file "$REPO/gp_env.yml" --prefix "$PREFIX"
        return
    fi
    python=${CONDA_PYTHON_EXE:-$(dirname -- "$(command -v conda)")/python}
    echo "this conda has no working 'conda env' subcommand, going through $python instead" >&2
    [[ -x $python ]] || die "no python next to conda; the conda installation is incomplete"
    "$python" -m conda_env.cli.main create --file "$REPO/gp_env.yml" --prefix "$PREFIX"
}

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
    [[ -n $ROOT ]] || die "no ReismanLab group folder found; pass --root /path/to/shared/folder"
fi

ENVS_DIR=$ROOT/envs
PKGS_DIR=$ROOT/pkgs
PREFIX=$ENVS_DIR/gp_env

command -v conda >/dev/null 2>&1 || die "conda is not on PATH; source the conda.sh of your installation first"
[[ -f $REPO/gp_env.yml ]] || die "gp_env.yml not found next to this script ($REPO)"

step "shared folders under $ROOT"
if [[ ! -d $ENVS_DIR || ! -d $PKGS_DIR ]]; then
    mkdir -p "$ENVS_DIR" "$PKGS_DIR" || die "cannot create $ENVS_DIR and $PKGS_DIR"
fi
# Group members create their environments and cache packages here too, and g+s keeps the group on
# whatever lands in the folders. The environment itself stays read-only for the group (below), so
# that a stray "pip install" cannot break it for everybody.
# Two calls on purpose: chmod applies a mode in one go, and where the setgid bit is refused the
# group permissions would be lost with it. Both fail harmlessly if someone else owns the folders.
chmod g+rwx "$ENVS_DIR" "$PKGS_DIR" 2>/dev/null || true
chmod g+s   "$ENVS_DIR" "$PKGS_DIR" 2>/dev/null || true
chmod g+rx "$ROOT" 2>/dev/null || true
printf 'environments: %s\npackage cache: %s\n' "$ENVS_DIR" "$PKGS_DIR"

if ((ALL_USERS)); then
    CONDARC=$ROOT/.condarc
    [[ -e $ROOT/etc/profile.d/conda.sh ]] || cat >&2 <<EOF
warning: $ROOT is not a conda installation, so nobody's conda reads
         $ROOT/.condarc. Either install miniconda there, or drop
         --all-users and have every group member run this script once.
EOF
else
    CONDARC=$HOME/.condarc
fi
step "conda configuration in $CONDARC"
# Touch first: in a file it creates itself, conda also writes a "channels: defaults" block.
[[ -e $CONDARC ]] || touch "$CONDARC" || die "cannot write $CONDARC"
conda config --file "$CONDARC" --prepend envs_dirs "$ENVS_DIR"
conda config --file "$CONDARC" --prepend pkgs_dirs "$PKGS_DIR"
# "already in the list, moving to the top" on a second run is the expected no-op.

# Check what conda really sees now, rather than what we asked for: the environment below is created
# by path and works either way, but activating it by name does need this.
if conda config --show envs_dirs 2>/dev/null | grep -qF -- "$ENVS_DIR"; then
    echo "envs_dirs and pkgs_dirs now start with the group folders"
else
    cat >&2 <<EOF
warning: conda does not see the new setting ("conda config --show envs_dirs" does not list
         $ENVS_DIR). "conda activate gp_env" by name will not work
         until it does; put this at the top of $CONDARC by hand:

envs_dirs:
  - $ENVS_DIR
pkgs_dirs:
  - $PKGS_DIR
EOF
fi

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
else
    # Only the member who builds the environment needs write access to the group folder.
    [[ -w $ENVS_DIR && -w $PKGS_DIR ]] || die "no write permission on $ENVS_DIR or $PKGS_DIR; ask whoever owns them to run this script, or to give the group write access"
    if ! env_create; then
        # Recent conda versions refuse to touch repo.anaconda.com before its Terms of Service have
        # been accepted once per account. Accepting them is your decision, not this script's.
        cat >&2 <<'EOF'

If the error above is about Terms of Service, run the two "conda tos accept" commands conda
printed, then run this script again; it continues where it stopped.
EOF
        exit 1
    fi
    # Belt and braces next to the umask: conda keeps the mode of files that come from the package
    # cache, and one unreadable file is enough to break the environment for the rest of the group.
    chmod -R g+rX "$PREFIX" 2>/dev/null || true
fi

if ((RUN_TESTS)); then
    step "test suite"
    (cd "$REPO" && conda run --prefix "$PREFIX" --no-capture-output pytest tests -m "not slow")
fi

cat <<EOF

Done. gp_env lives in $PREFIX; the only file changed outside it is $CONDARC.
Activate it from anywhere, by name:

    conda activate gp_env

Every other member of the group gets the same environment with the same command, which then
only edits their own ~/.condarc and takes a few seconds:

    bash setup_gp_env.sh --no-test
EOF
