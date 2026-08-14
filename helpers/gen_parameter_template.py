#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Regenerate ``helpers/parameterFile.prm`` from ``utils/dftParameters.cc``.

The socket driver seeds every run from that template and then overwrites
individual entries. If a key is absent from the template the injector has to
invent it -- and inventing a *nested* subsection at top level makes deal.II
silently drop the whole block (``parse_input`` runs with
``skip_undefined=true``). Emitting every declared key at its declared default,
with the declared nesting, removes that failure mode: injection only ever
replaces a line that is already there.

Run after touching ``declare_parameters`` in ``dftParameters.cc``:

    python3 helpers/gen_parameter_template.py
"""

from __future__ import annotations

import os
import re
import sys

# Solver-mode blocks the socket driver never runs; omitted to keep the template
# readable. Nothing injects into them.
SKIP_SUBSECTIONS = {"FunctionalTest", "Molecular Dynamics"}

HEADER = """\
# AUTO-GENERATED -- do not edit by hand.
# Source of truth: utils/dftParameters.cc declare_parameters().
# Regenerate: python3 helpers/gen_parameter_template.py
#
# Every declared parameter appears here at its DFT-FE default, with the same
# nesting as declare_parameters(). The socket driver (dftfeWrapper::reinit)
# copies this file and REPLACES entries in place -- it never has to create a
# subsection, so a nested block can never be emitted at top level and silently
# dropped by parse_input(skip_undefined=true).
#
# SOLVER MODE / VERBOSITY are read by utils/runParameters.cc (top level), not by
# dftParameters; they are kept here so the template also works as a native deck.
set SOLVER MODE=GS
set VERBOSITY=1
"""

TOKEN = re.compile(
    r'prm\.enter_subsection\(\s*"([^"]+)"\s*\)'
    r'|prm\.leave_subsection\(\)'
    r'|prm\.declare_entry\(\s*("(?:[^"\\]|\\.)*")\s*,\s*("(?:[^"\\]|\\.)*")'
)


def generate(source: str) -> str:
    # Only declare_parameters(); parse_parameters() re-enters every subsection to
    # read values back and would otherwise emit a second, empty copy of each.
    start = source.index("declare_parameters(dealii::ParameterHandler &prm)")
    end = source.index("dftParameters::parse_parameters", start)
    body = source[start:end]
    out, depth, skip_depth = [], 0, None

    for tok in TOKEN.finditer(body):
        name = tok.group(1)
        if name:                                   # enter_subsection
            depth += 1
            if skip_depth is None and name in SKIP_SUBSECTIONS:
                skip_depth = depth
            elif skip_depth is None:
                out.append("  " * (depth - 1) + f"subsection {name}")
        elif tok.group(0).startswith("prm.leave"):  # leave_subsection
            if skip_depth == depth:
                skip_depth = None
            elif skip_depth is None:
                out.append("  " * (depth - 1) + "end")
            depth -= 1
        elif skip_depth is None:                   # declare_entry
            key = tok.group(2)[1:-1]
            default = tok.group(3)[1:-1].replace('\\"', '"')
            out.append("  " * depth + f"set {key}={default}")

    if depth != 0:
        raise SystemExit(f"unbalanced subsections in dftParameters.cc (depth {depth})")
    return HEADER + "\n".join(out) + "\n"


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "utils", "dftParameters.cc")
    dst = os.path.join(root, "helpers", "parameterFile.prm")
    text = generate(open(src).read())
    n = sum(1 for line in text.splitlines() if line.lstrip().startswith("set "))
    with open(dst, "w") as fh:
        fh.write(text)
    print(f"wrote {dst} ({n} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
