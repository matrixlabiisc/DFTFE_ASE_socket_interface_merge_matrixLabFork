# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""The "unset kwarg => DFT-FE's own default" invariant.

A parameter the user never mentions must reach the solver at exactly the value
`dftParameters.cc` declares for it. Nothing in Python may invent a value, and
nothing in C++ may inject one. That holds only if three things stay true, and
each is a separate failure mode that no runtime test would catch (the run
succeeds -- it just silently uses a different number):

  1. `helpers/parameterFile.prm`, the template every socket run is seeded from,
     still equals the declared defaults. It is generated, so it goes stale the
     moment someone edits `declare_parameters()` without rerunning the script.
  2. Every physics parameter in `dftfeWrapper::reinit` is sentinel-guarded, so
     "not sent" never turns into "injected as -1".
  3. The sentinel `socket_interface.cc` substitutes for an absent JSON field is
     the same sentinel the wrapper's guard tests against. A mismatch here means
     an unset parameter is injected as its sentinel.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from dftfe_ase.params import to_prm_entries

DFTFE_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_CC = DFTFE_ROOT / "src" / "dftfeWrapper.cc"
SOCKET_CC = DFTFE_ROOT / "src" / "socket_interface.cc"
GEN_SCRIPT = DFTFE_ROOT / "helpers" / "gen_parameter_template.py"
TEMPLATE = DFTFE_ROOT / "helpers" / "parameterFile.prm"

needs_cxx = pytest.mark.skipif(
    not WRAPPER_CC.exists(),
    reason="C++ sources not present (installed package, not a source checkout)",
)

# Arguments of reinit() that are *meant* to be applied unconditionally: geometry
# and cell come from the Atoms object on every call, and the rest are decisions
# the driver owns rather than parameters the user may leave unset.
DRIVER_OWNED_ARGS = {
    "mpi_comm_parent",
    "useDevice",
    "atomicPositionsCart",
    "atomicNumbers",
    "cell",
    "pbc",
    "mpGrid",
    "mpGridShift",
    "pseudopotentialFilename",
    "setDeviceToMPITaskBindingInternally",
    "keepScratch",
    "computeIonForces",
    "computeStress",
}


def _reinit_body() -> str:
    """Source of the full reinit() overload, brace-matched."""
    src = WRAPPER_CC.read_text()
    start = None
    for m in re.finditer(r"dftfeWrapper::reinit\(", src):
        head = src[m.end() : m.end() + 4000]
        close = head.index(")\n  {")
        if "computeStress" in head[:close]:
            start = m
            sig = head[:close]
            break
    assert start is not None, "full reinit() overload not found"

    i = src.index("{", start.end() + close)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return sig, src[i : j + 1]
    raise AssertionError("unbalanced braces in reinit()")


def test_template_still_matches_declared_defaults():
    """Regenerating the template from dftParameters.cc must be a no-op."""
    if not GEN_SCRIPT.exists():
        pytest.skip("generator not present")
    spec = importlib.util.spec_from_file_location("gen_prm", GEN_SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    expected = gen.generate((DFTFE_ROOT / "utils" / "dftParameters.cc").read_text())
    assert TEMPLATE.read_text() == expected, (
        "helpers/parameterFile.prm is stale: it no longer matches the defaults "
        "declared in dftParameters.cc. Every unset parameter would silently run "
        "at the template's value instead. Fix: python3 helpers/gen_parameter_template.py"
    )


@needs_cxx
def test_every_physics_parameter_is_sentinel_guarded():
    """No reinit() argument may be injected without first testing its sentinel."""
    sig, body = _reinit_body()
    args = set(re.findall(r"[\w:<>]+\s+&?(\w+)\s*(?=,|$)", sig))
    guarded = set(re.findall(r"if \((\w+) !=", body))

    unguarded = args - guarded - DRIVER_OWNED_ARGS
    assert not unguarded, (
        f"reinit() arguments injected without a sentinel guard: {sorted(unguarded)}. "
        "An unset kwarg would be written into the .prm as its sentinel value "
        "(-1 / 999999 / 'Unprovided') instead of leaving DFT-FE's default alone."
    )


@needs_cxx
def test_request_defaults_match_wrapper_sentinels():
    """The value used for an absent JSON field must be the value guarded on."""
    _, body = _reinit_body()
    socket = SOCKET_CC.read_text()

    # socket_interface.cc: parse_scalar<T>(json, "name", DEFAULT)
    defaults = dict(
        re.findall(r'parse_(?:scalar<[^>]+>|string)\(json,\s*"(\w+)",\s*([^)]+)\)', socket)
    )
    sentinels = {v.strip() for v in defaults.values()}
    sentinels.discard("false")  # keep_scratch: a real default, not a sentinel

    # Only guards that actually wrap an injection; the function also contains
    # ordinary control flow (mpi_comm_parent != MPI_COMM_NULL, and so on).
    guarded_against = {
        m.group(1)
        for m in re.finditer(r"if \(\w+ != ([^)]+)\)", body)
        if "applyPrm" in body[m.end() : m.end() + 400]
    }
    stray = {s for s in guarded_against if s not in sentinels}
    assert not stray, (
        f"wrapper guards against value(s) no request field ever defaults to: {sorted(stray)}. "
        "The guard can never fire, so the parameter is always injected."
    )


def test_unset_kwarg_produces_no_prm_entry():
    """Python side: a None kwarg is dropped, not formatted."""
    assert to_prm_entries({"mesh_size": None, "atom_ball_radius": None}) == []
    assert to_prm_entries({"mem_opt_mode": None}) == []


def test_falsy_but_set_values_are_still_injected():
    """0 / False are user intent, not absence -- they must survive the None filter."""
    assert to_prm_entries({"adapt_anderson_mixing_parameter": False}) == [
        {"section": "SCF parameters",
         "key": "ADAPT ANDERSON MIXING PARAMETER",
         "value": "false"}
    ]
    assert to_prm_entries({"scalapack_procs": 0})[0]["value"] == 0
