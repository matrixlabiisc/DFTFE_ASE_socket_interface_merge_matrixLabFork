#!/bin/bash
# ---------------------------------------------------------------------
# Submit one arm of the BCC-Mo 431-atom scaling study.
#
#   ./submit.sh native 32
#   ./submit.sh ase    32
#
# PBS needs the node count at submit time (-l select), and the job scripts need
# it at run time (NODES), so both are set here from one argument -- they cannot
# be allowed to disagree.
#
# debug-scaling allows ONE queued job per user, so run these one at a time.
# ---------------------------------------------------------------------

set -euo pipefail

ARM="${1:-}"
NODES="${2:-}"

if [[ "$ARM" != "native" && "$ARM" != "ase" ]] || [[ -z "$NODES" ]]; then
    echo "usage: $0 <native|ase> <nodes>" >&2
    exit 2
fi

: "${DFTFE_INSTALL:?export DFTFE_INSTALL=<ase_redesign install prefix>}"
: "${ASE_DFTFE_SRC:?export ASE_DFTFE_SRC=<.../interfaces/ase-dftfe/src>}"

SCRIPT="run_${ARM}.pbs"
[[ -f "$SCRIPT" ]] || { echo "missing $SCRIPT" >&2; exit 1; }

PASS="NODES=${NODES},DFTFE_INSTALL=${DFTFE_INSTALL},ASE_DFTFE_SRC=${ASE_DFTFE_SRC}"
[[ -n "${PYTHON:-}" ]] && PASS="${PASS},PYTHON=${PYTHON}"

set -x
qsub -l select="${NODES}" \
     -N "bccmo431_${ARM}_${NODES}n" \
     -v "$PASS" \
     "$SCRIPT"
