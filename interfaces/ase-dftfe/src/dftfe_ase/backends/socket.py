# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Socket backend: Python is the TCP server, DFT-FE the launched client.

Lifecycle: bind a server socket -> launch DFT-FE (``<command> --socket host:port``)
-> accept the connection and read the ``READY`` handshake -> exchange one
request/response per ``compute()`` -> send ``exit`` and reap on ``close()``.

This module fixes the legacy calculator's rough edges: the ``verbosity is None``
crash, the fragile response framing, ``shell=True`` launching, the bare
``except``, and the reliance on ``__del__`` for cleanup.
"""

from __future__ import annotations

import logging
import os
import shlex
import socket
import subprocess
import time

from .base import Backend
from ..protocol import HANDSHAKE, CMD_EXIT, MessageChannel, ProtocolError

log = logging.getLogger("dftfe_ase.socket")


class DFTFEError(RuntimeError):
    """Raised when DFT-FE fails to start, dies, or reports an error."""


class SocketBackend(Backend):
    def __init__(
        self,
        command: str = "dftfe",
        host: str | None = None,  # deprecated alias for advertise_host
        port: int = 0,
        *,
        bind_host: str = "0.0.0.0",
        advertise_host: str | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        connect_timeout: float = 120.0,
        log_file: str = "dftfe.log",
        verbosity: int | None = None,
    ):
        self.command = command
        # bind_host: interface the server listens on. 0.0.0.0 accepts connections
        #   from any node (required for multi-node, where DFT-FE rank 0 may be on
        #   a different node than this Python process).
        # advertise_host: the address DFT-FE is told to connect back to. It MUST
        #   be routable from the compute nodes -> the node hostname, NOT 127.0.0.1
        #   (localhost would make an off-node rank 0 connect to itself and hang).
        self.bind_host = bind_host
        self.advertise_host = advertise_host or host or socket.gethostname()
        self.port = port
        self.env = env
        self.cwd = cwd
        self.connect_timeout = connect_timeout
        self.log_file = log_file
        self.verbosity = verbosity

        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._chan: MessageChannel | None = None
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._started = False
        self.last_wait_s = None  # wall time of the last socket round-trip (send->recv)
        self.startup_s = None    # process launch + MPI init + connect (one-time)
        self.bind_s = 0.0        # TCP server setup            (interface-only)
        self.spawn_s = 0.0       # fork mpiexec                (native pays too)
        self.boot_s = 0.0        # DFT-FE MPI_Init + start     (native pays too)
        self.handshake_s = 0.0   # READY line                  (interface-only)

    # ── helpers ─────────────────────────────────────────────────────────
    def _verbose(self) -> bool:
        # NOTE: guarded against None — the legacy bug was `if self.verbosity > 0`.
        return self.verbosity is not None and self.verbosity > 0

    def _preflight_advertise(self) -> None:
        # The advertised host must be resolvable so DFT-FE rank 0 can connect
        # back. Fail fast with a clear message rather than a mysterious hang.
        try:
            socket.getaddrinfo(self.advertise_host, self.port or None,
                               proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise DFTFEError(
                f"advertise_host {self.advertise_host!r} does not resolve; DFT-FE "
                f"would fail to connect back. Set advertise_host to a hostname/IP "
                f"routable from the compute nodes."
            ) from exc

    def _start_server(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.bind_host, self.port))
        self.port = self._server.getsockname()[1]  # resolve if port was 0
        self._server.listen(1)
        if self._verbose():
            log.info("listening on %s:%d, advertising %s:%d to DFT-FE",
                     self.bind_host, self.port, self.advertise_host, self.port)

    def _launch(self) -> None:
        self._preflight_advertise()
        base = self.command if isinstance(self.command, (list, tuple)) else shlex.split(self.command)
        argv = list(base) + ["--socket", f"{self.advertise_host}:{self.port}"]
        env = None
        if self.env:
            env = dict(os.environ)
            env.update(self.env)
        log_path = self.log_file
        if self.cwd:
            os.makedirs(self.cwd, exist_ok=True)
            log_path = os.path.join(self.cwd, os.path.basename(self.log_file))
        self._log_fh = open(log_path, "w", buffering=1)
        if self._verbose():
            log.info("launching: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv, stdout=self._log_fh, stderr=subprocess.STDOUT, cwd=self.cwd, env=env
        )

    def _accept(self) -> None:
        self._server.settimeout(self.connect_timeout)
        t0 = time.perf_counter()
        try:
            self._conn, _addr = self._server.accept()
        except socket.timeout as exc:
            code = self._proc.poll() if self._proc else None
            self.close()
            raise DFTFEError(
                f"DFT-FE did not connect within {self.connect_timeout}s "
                f"(process exit code {code}); see {self.log_file}"
            ) from exc
        # Time spent blocked here is DFT-FE booting: MPI_Init across every rank
        # plus program start. A native run pays exactly the same cost; it simply
        # is not visible because DFT-FE's internal timer starts after MPI_Init.
        self.boot_s = time.perf_counter() - t0
        self._conn.settimeout(None)  # SCF can be arbitrarily slow
        t1 = time.perf_counter()
        self._chan = MessageChannel(self._conn)
        handshake = self._chan.recv_handshake()
        self.handshake_s = time.perf_counter() - t1
        if handshake != HANDSHAKE:
            log.warning("unexpected handshake from DFT-FE: %r", handshake)

    # ── Backend API ─────────────────────────────────────────────────────
    def start(self) -> None:
        """Bring up the server and the DFT-FE process.

        Timed in pieces so the caller can separate cost the interface *adds*
        from cost a native DFT-FE run pays anyway:

          bind_s      -- create/bind/listen the TCP server      (interface only)
          spawn_s     -- fork mpiexec                           (native does this too)
          boot_s      -- block until DFT-FE connects: MPI_Init
                         across all ranks + program start       (native does this too)
          handshake_s -- read the READY line                    (interface only)
        """
        if self._started:
            return
        t0 = time.perf_counter()
        self._start_server()
        self.bind_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        self._launch()          # spawn mpirun + DFT-FE (MPI init happens here)
        self.spawn_s = time.perf_counter() - t1
        self._accept()          # sets boot_s and handshake_s
        self.startup_s = time.perf_counter() - t0
        self._started = True

    def compute(self, request: dict) -> dict:
        if not self._started:
            self.start()
        if self._proc is not None and self._proc.poll() is not None:
            raise DFTFEError(
                f"DFT-FE process exited (code {self._proc.returncode}); see {self.log_file}"
            )
        try:
            t0 = time.perf_counter()
            self._chan.send(request)
            response = self._chan.recv()
            self.last_wait_s = time.perf_counter() - t0
            return response
        except (ProtocolError, OSError) as exc:
            code = self._proc.poll() if self._proc else None
            raise DFTFEError(
                f"lost connection to DFT-FE (process exit code {code}); see {self.log_file}"
            ) from exc

    def close(self) -> None:
        if self._chan is not None:
            try:
                self._chan.send({"cmd": CMD_EXIT})
            except OSError:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None
        self._chan = None
        self._started = False
