# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""A fake DFT-FE binary for testing — speaks the socket protocol, no physics.

Launched exactly like the real binary (``mock_dftfe.py --socket host:port``),
it connects back to the Python server, sends the ``READY`` handshake, then
answers each request with deterministic, unit-checkable values:

    energy (Ha)      = -1.0 * sum(atomic_numbers)
    forces (Ha/Bohr) = atom i -> [0.001*(i+1), 0.002*(i+1), -0.003*(i+1)]
    stress (Ha/Bohr^3) = diag(0.01, 0.02, 0.03)

Framing is hand-rolled (stdlib only) on purpose: an independent implementation
of the wire format is a stronger test double than reusing the code under test.

Test hooks via environment variables:
    MOCK_DFTFE_ERROR=1   -> reply with an {"error": ...} frame
    MOCK_DFTFE_NOCONNECT=1 -> exit immediately without connecting
"""

import json
import os
import socket
import sys


def parse_socket_arg(argv):
    for i, a in enumerate(argv):
        if a == "--socket" and i + 1 < len(argv):
            host, port = argv[i + 1].rsplit(":", 1)
            return host, int(port)
    raise SystemExit("mock_dftfe: missing --socket host:port")


def main():
    if os.environ.get("MOCK_DFTFE_NOCONNECT") == "1":
        return
    host, port = parse_socket_arg(sys.argv[1:])
    sock = socket.create_connection((host, port), timeout=30)
    sock.sendall(b"READY\n")

    buf = bytearray()

    def read_line():
        while b"\n" not in buf:
            data = sock.recv(65536)
            if not data:
                return None
            buf.extend(data)
        nl = buf.index(b"\n")
        line = bytes(buf[:nl])
        del buf[: nl + 1]
        return line

    while True:
        line = read_line()
        if line is None:
            break
        req = json.loads(line.decode("utf-8"))
        if req.get("cmd") == "exit":
            break

        if os.environ.get("MOCK_DFTFE_ERROR") == "1":
            sock.sendall((json.dumps({"error": "mock failure"}) + "\n").encode())
            continue

        numbers = req.get("numbers", [])
        n = len(numbers)
        energy = -1.0 * sum(numbers)
        resp = {"energy": energy}
        if req.get("compute_forces", True):
            resp["forces"] = [
                [0.001 * (i + 1), 0.002 * (i + 1), -0.003 * (i + 1)] for i in range(n)
            ]
        if req.get("compute_stress", False):
            resp["stress"] = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]]
        resp["compute_time"] = 0.0  # mock does no real compute
        sock.sendall((json.dumps(resp) + "\n").encode())

    sock.close()


if __name__ == "__main__":
    main()
