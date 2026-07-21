# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Wire protocol for the ASE <-> DFT-FE socket interface.

Single source of truth for framing and message schema, mirrored by the C++
``socketDriver``. Protocol v1 uses newline-delimited JSON, matching the deployed
DFT-FE build. A binary length-prefixed framing (v2) can replace the framing in
:class:`MessageChannel` behind the same ``send``/``recv`` API without touching
callers (the calculator and backend never parse bytes directly).

Flow (Python is the server, DFT-FE the client):
    1. DFT-FE connects and sends the handshake line ``READY``.
    2. Python sends a request object; DFT-FE replies with a response object.
    3. Repeat for each ``calculate()``.
    4. Python sends ``{"cmd": "exit"}`` to shut the server down.
"""

from __future__ import annotations

import json

PROTOCOL_VERSION = 1
HANDSHAKE = "READY"
CMD_RUN = "run"
CMD_EXIT = "exit"


class ProtocolError(RuntimeError):
    """Raised on framing/parse errors or an unexpectedly closed connection."""


def encode_message(obj: dict) -> bytes:
    """Serialize a message object to a single newline-terminated JSON frame."""
    return (json.dumps(obj) + "\n").encode("utf-8")


class MessageChannel:
    """Newline-delimited JSON framing over a connected stream socket.

    Buffers partial reads, so a message split across TCP segments — or several
    messages arriving in one segment — are handled correctly. This is the fix
    for the legacy "break on first ``}``" bug that truncated large force arrays.
    """

    def __init__(self, sock, chunk_size: int = 65536):
        self._sock = sock
        self._chunk = chunk_size
        self._buf = bytearray()

    def _read_line(self) -> bytes:
        while b"\n" not in self._buf:
            data = self._sock.recv(self._chunk)
            if not data:
                raise ProtocolError("connection closed by peer while reading")
            self._buf.extend(data)
        nl = self._buf.index(b"\n")
        line = bytes(self._buf[:nl])
        del self._buf[: nl + 1]
        return line

    def recv_handshake(self) -> str:
        """Read the plain-text handshake line (not JSON)."""
        return self._read_line().decode("utf-8").strip()

    def send(self, obj: dict) -> None:
        self._sock.sendall(encode_message(obj))

    def recv(self) -> dict:
        line = self._read_line()
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON message: {line!r}") from exc
