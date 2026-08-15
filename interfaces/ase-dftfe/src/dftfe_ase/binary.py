# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""DFT-FE binary selection.

Two orthogonal facts about "which binary":
  * **kind** (real vs complex) is a *physics* decision made from the k-point
    sampling — real for Gamma-only, complex when k-points are sampled. Decided
    here in Python from the ASE inputs.
  * **location** (the file paths) is *configuration* — resolved in
    :mod:`dftfe_ase.config` from kwargs/env/cluster-profile/container/PATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

REAL = "real"
COMPLEX = "complex"


class DFTFEConfigError(RuntimeError):
    """Raised when the DFT-FE binary/cluster configuration cannot be resolved."""


def select_binary_kind(
    mp_grid: Optional[Sequence[int]] = None,
    mp_grid_shift: Optional[Sequence[float]] = None,
    pbc: Optional[Sequence[bool]] = None,
) -> str:
    """Return ``"real"`` (Gamma-only) or ``"complex"`` (k-point sampling).

    Rules: a non-periodic system is Gamma-only -> real. A 1x1x1 unshifted grid
    is Gamma-only -> real. Any genuine k-point mesh (or a shift) -> complex.
    """
    if pbc is not None and not any(bool(p) for p in pbc):
        return REAL
    if mp_grid is None:
        return REAL
    grid = tuple(int(x) for x in mp_grid)
    if grid == (1, 1, 1):
        shift = tuple(mp_grid_shift) if mp_grid_shift is not None else (0, 0, 0)
        if all(float(s) == 0.0 for s in shift):
            return REAL
    return COMPLEX


@dataclass
class DFTFEBinaries:
    """Resolved DFT-FE executable paths for each kind."""

    real: Optional[str] = None
    complex: Optional[str] = None

    def path_for(self, kind: str) -> str:
        path = self.real if kind == REAL else self.complex
        if not path:
            raise DFTFEConfigError(
                f"no {kind!r} DFT-FE binary configured. Binary paths are per-user, "
                f"so no cluster profile ships one. Give it as dftfe_{kind}=..., or "
                f"bin_dir=..., or set DFTFE_BIN_{kind.upper()} / DFTFE_BIN_DIR, or "
                f"put {{\"<cluster>\": {{\"bin_dir\": \"...\"}}}} in "
                f"~/.config/dftfe_ase/profiles.json"
            )
        return path
