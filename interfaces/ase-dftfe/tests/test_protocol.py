# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for the wire protocol framing (no ASE, no subprocess)."""

import socket
import threading

import pytest

from dftfe_ase.protocol import MessageChannel, ProtocolError, encode_message


def test_encode_message_is_newline_terminated_json():
    raw = encode_message({"a": 1, "b": [1, 2, 3]})
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1


def test_roundtrip_over_socketpair():
    a, b = socket.socketpair()
    try:
        ca, cb = MessageChannel(a), MessageChannel(b)
        ca.send({"cmd": "run", "coords": [[0.0, 0.0, 0.0]]})
        assert cb.recv() == {"cmd": "run", "coords": [[0.0, 0.0, 0.0]]}
    finally:
        a.close()
        b.close()


def test_message_split_across_chunks_is_reassembled():
    """The legacy bug: breaking on the first '}' truncated multi-chunk frames."""
    a, b = socket.socketpair()
    try:
        payload = encode_message({"forces": [[1.0, 2.0, 3.0]] * 500})
        mid = len(payload) // 2

        def sender():
            a.sendall(payload[:mid])
            a.sendall(payload[mid:])

        t = threading.Thread(target=sender)
        t.start()
        msg = MessageChannel(b).recv()
        t.join()
        assert len(msg["forces"]) == 500
    finally:
        a.close()
        b.close()


def test_two_messages_in_one_segment():
    a, b = socket.socketpair()
    try:
        a.sendall(encode_message({"i": 1}) + encode_message({"i": 2}))
        chan = MessageChannel(b)
        assert chan.recv() == {"i": 1}
        assert chan.recv() == {"i": 2}
    finally:
        a.close()
        b.close()


def test_closed_connection_raises_protocol_error():
    a, b = socket.socketpair()
    a.close()
    with pytest.raises(ProtocolError):
        MessageChannel(b).recv()
    b.close()


def test_handshake_line_then_json_share_buffer():
    a, b = socket.socketpair()
    try:
        a.sendall(b"READY\n" + encode_message({"ok": True}))
        chan = MessageChannel(b)
        assert chan.recv_handshake() == "READY"
        assert chan.recv() == {"ok": True}
    finally:
        a.close()
        b.close()
