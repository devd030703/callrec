"""Persistent, disposable model child with a deadline for each complete window."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import resource
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from .common import CallrecError


def model_child(connection, engine, model, log_path):
    parent = os.getppid()

    def parent_watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)  # Also handles a supervisor killed while native inference is stuck.

    threading.Thread(target=parent_watch, daemon=True).start()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    blocked = []
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(method):
        def connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                blocked.append(True)
                raise RuntimeError(
                    "Callrec model inference is offline; network connection blocked."
                )
            return method(sock, address)

        return connect

    socket.socket.connect = guard(original_connect)
    socket.socket.connect_ex = guard(original_connect_ex)

    def statistics():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_rss_mib": round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1),
            "blocked_network_attempts": len(blocked),
            "network_guard": True,
        }

    try:
        with open(log_path, "a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if engine in ("qwen", "qwen-large"):
                    from mlx_audio.stt import load

                    backend = load(model, strict=True).generate
                else:
                    from mlx_whisper import transcribe

                    backend = transcribe
                while True:
                    request = connection.recv()
                    if request is None:
                        break
                    samples, options = request
                    try:
                        output = backend(samples, **options)
                        if engine in ("qwen", "qwen-large"):
                            output = {
                                k: getattr(output, k)
                                for k in ("text", "language", "generation_tokens")
                            }
                        connection.send((True, output, statistics()))
                    except Exception:
                        connection.send((False, traceback.format_exc(), statistics()))
    except EOFError:
        pass
    except Exception:
        try:
            connection.send((False, traceback.format_exc(), statistics()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ModelProcess:
    def __init__(self, engine, model, log_path, event=None, timeout=180, target=model_child):
        self.engine, self.model, self.log_path = engine, str(model), str(log_path)
        self.event = event or (lambda *a, **k: None)
        self.timeout, self.target = timeout, target
        self.process = self.connection = None
        self.deadline = None
        self.wall_deadline = None
        self.stats = {}

    def start(self):
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(
            target=self.target,
            args=(child, self.engine, self.model, self.log_path),
            daemon=True,
        )
        self.process.start()
        child.close()
        self.event("inference_started", pid=self.process.pid)

    def close(self, force=False):
        if self.process is not None:
            if not force and self.process.is_alive():
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
            if self.process.is_alive():
                self.process.kill()
            self.process.join(3)
            self.event("inference_stopped", pid=self.process.pid)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def window(self, operation):
        for attempt in range(2):
            self.deadline = time.monotonic() + self.timeout
            self.wall_deadline = time.time() + self.timeout
            self.event("inference_window_started", retry=attempt == 1)
            try:
                return operation()
            except TimeoutError:
                self.event("window_timeout", retry=attempt == 0, deadline_seconds=self.timeout)
                self.close(force=True)
                if attempt:
                    raise CallrecError(
                        "Audio window exceeded 180-second inference deadline twice; "
                        "audio and completed checkpoints retained. Retry this session."
                    )
        raise AssertionError("unreachable")

    def __call__(self, samples, **options):
        if self.process is None:
            self.start()
        try:
            self.connection.send((samples, options))
            while True:
                remaining = min(self.deadline - time.monotonic(), self.wall_deadline - time.time())
                if remaining <= 0:
                    raise TimeoutError("Inference window deadline exceeded")
                if self.connection.poll(min(0.5, remaining)):
                    break
            ok, output, stats = self.connection.recv()
            self.stats = {
                **stats,
                "peak_rss_mib": max(
                    stats.get("peak_rss_mib", 0), self.stats.get("peak_rss_mib", 0)
                ),
            }
        except TimeoutError:
            raise
        except (EOFError, BrokenPipeError, OSError) as e:
            raise CallrecError("Inference child exited; audio/checkpoints retained.") from e
        if not ok:
            raise CallrecError(f"Inference child failed: {output[-2000:]}")
        return SimpleNamespace(**output) if self.engine in ("qwen", "qwen-large") else output
