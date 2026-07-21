# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for real/complex binary selection."""

import pytest

from dftfe_ase.binary import (
    COMPLEX,
    REAL,
    DFTFEBinaries,
    DFTFEConfigError,
    select_binary_kind,
)


@pytest.mark.parametrize(
    "mp_grid, shift, pbc, expected",
    [
        (None, None, None, REAL),                       # default -> gamma
        (None, None, [False, False, False], REAL),      # non-periodic
        ((1, 1, 1), (0, 0, 0), [True, True, True], REAL),   # gamma unshifted
        ((1, 1, 1), None, [True, True, True], REAL),
        ((2, 2, 2), (0, 0, 0), [True, True, True], COMPLEX),  # k-mesh
        ((1, 1, 1), (1, 1, 1), [True, True, True], COMPLEX),  # shifted gamma
        ((4, 4, 1), None, [True, True, False], COMPLEX),      # slab k-mesh
        ((2, 2, 2), None, [False, False, False], REAL),       # pbc off overrides
    ],
)
def test_select_binary_kind(mp_grid, shift, pbc, expected):
    assert select_binary_kind(mp_grid, shift, pbc) == expected


def test_binaries_path_for():
    b = DFTFEBinaries(real="/r/dftfe", complex="/c/dftfe")
    assert b.path_for(REAL) == "/r/dftfe"
    assert b.path_for(COMPLEX) == "/c/dftfe"


def test_binaries_missing_raises():
    with pytest.raises(DFTFEConfigError):
        DFTFEBinaries().path_for(REAL)
