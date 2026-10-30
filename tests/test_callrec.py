import contextlib
import hashlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from callrec.cli import main, stop_worker
from callrec.common import (
    HARK_SHA256,
    HARK_VERSION,
    MODEL_REVISION,
    CallrecError,
    atomic_json,
    create_session,
    lock,
    read_json,
    sha256,
    update_session,
)
from callrec.setup import download_model_file
from callrec.transcription import (
    owned_segments,
    prepare_audio,
    qwen_windows,
    transcribe_session,
    windows,
)
from callrec.worker import ChunkWriter, capture


def wav(path, seconds=1, rate=16000, channels=1):
    with wave.open(str(path), "wb") as f:
        f.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        f.writeframes(b"\x00\x00" * round(rate * seconds) * channels)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_setup_resumes_exact_range_and_verifies_digest(self):
        payload = b"abcdef"
        (self.root / "fixture.bin.download").write_bytes(payload[:3])
        response = io.BytesIO(payload[3:])
        response.status = 206
        response.headers = {"Content-Range": "bytes 3-5/6"}
        expected = {"fixture.bin": hashlib.sha256(payload).hexdigest()}
        with (
            patch("callrec.setup.MODEL_SHA256", expected),
            patch("callrec.setup.urllib.request.urlopen", return_value=response) as request,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            download_model_file(self.root, "fixture.bin", len(payload))
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-5")
        self.assertEqual((self.root / "fixture.bin").read_bytes(), payload)
        self.assertFalse((self.root / "fixture.bin.download").exists())

    def test_setup_does_not_append_wrong_range_response(self):
        partial = self.root / "fixture.bin.download"
        partial.write_bytes(b"abc")
        response = io.BytesIO(b"abcdef")
        response.status = 200
        response.headers = {}
        with (
            patch("callrec.setup.MODEL_SHA256", {"fixture.bin": "unused"}),
            patch("callrec.setup.urllib.request.urlopen", return_value=response),
            self.assertRaises(CallrecError),
        ):
            download_model_file(self.root, "fixture.bin", 6)
        self.assertEqual(partial.read_bytes(), b"abc")

    def qwen_model(self):
        from callrec.common import model_details

        details = model_details("qwen")
        path = self.root / "models" / details["directory"]
        path.mkdir(parents=True, exist_ok=True)
        for name in details["files"]:
            (path / name).write_text("fixture")
        atomic_json(path / "callrec-model.json", {"revision": details["revision"]})

    def test_qwen_hindi_hint_preserves_mixed_script_and_marks_coarse_times(self):
        self.qwen_model()
        session = self.imported(seconds=50)
        update_session(session, engine="qwen")
        calls = []

        def backend(samples, **kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["language"], "Hindi")
            return SimpleNamespace(
                text="आज we discussed français", language=["Hindi"], generation_tokens=8
            )

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(session, self.root, backend=backend)
        report = read_json(session / "transcript.json")
        self.assertEqual(report["settings"]["engine"], "qwen")
        self.assertTrue(report["processing_complete"])
        self.assertEqual(report["segments"][-1]["end"], 50)
        self.assertEqual(report["segments"][0]["timestamp_resolution"], "audio_chunk")
        self.assertIn("आज", (session / "transcript.md").read_text())

    def test_qwen_token_limit_cannot_silently_complete(self):
        self.qwen_model()
        session = self.imported()
        update_session(session, engine="qwen")

        def exhausted(*args, **kwargs):
            return SimpleNamespace(text="partial", language=["Hindi"], generation_tokens=2048)

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(CallrecError):
            transcribe_session(session, self.root, backend=exhausted)
        self.assertEqual(read_json(session / "session.json")["status"], "transcription_failed")
        self.assertFalse((session / "transcript.md").exists())

    def test_qwen_token_recovery_is_bounded_and_auditable(self):
        self.qwen_model()
        session = self.imported()
        update_session(session, engine="qwen")
        calls = []

        def backend(samples, **kwargs):
            calls.append(kwargs)
            if "repetition_penalty" not in kwargs:
                return SimpleNamespace(text="loop", language=["Hindi"], generation_tokens=2048)
            return SimpleNamespace(
                text="सुबह चाय पीते हैं", language=["Hindi"], generation_tokens=12
            )

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(session, self.root, backend=backend)
        report = read_json(session / "transcript.json")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["max_tokens"], 2048)
        self.assertEqual(calls[1]["max_tokens"], 512)
        self.assertEqual(calls[1]["repetition_penalty"], 1.1)
        self.assertEqual(report["recovery_events"][0]["index"], 0)
        self.assertIn("सुबह चाय", report["segments"][0]["text"])
        self.assertEqual(read_json(session / "session.json")["status"], "complete_with_warnings")

    def test_qwen_recovery_resumes_without_rewriting_completed_checkpoint(self):
        self.qwen_model()
        session = self.imported(seconds=19)
        update_session(session, engine="qwen")
        first = True

        def interrupted(samples, **kwargs):
            nonlocal first
            if first:
                first = False
                return SimpleNamespace(text="पहला भाग", language=["Hindi"], generation_tokens=8)
            return SimpleNamespace(text="loop", language=["Hindi"], generation_tokens=2048)

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(CallrecError):
            transcribe_session(session, self.root, backend=interrupted)
        checkpoint = session / "checkpoints/000000.json"
        previous = checkpoint.read_bytes()
        calls = []

        def recovered(samples, **kwargs):
            calls.append(kwargs)
            count = 8 if "repetition_penalty" in kwargs else 2048
            return SimpleNamespace(text="दूसरा भाग", language=["Hindi"], generation_tokens=count)

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(session, self.root, backend=recovered)
        self.assertEqual(checkpoint.read_bytes(), previous)
        self.assertEqual(len(calls), 2)
        report = read_json(session / "transcript.json")
        self.assertEqual(report["processed_windows"], 2)
        self.assertEqual(report["recovery_events"][0]["index"], 1)

    def test_qwen_windows_cover_all_samples_without_overlap_or_gaps(self):
        path = self.root / "long.wav"
        wav(path, seconds=123)
        with wave.open(str(path)) as f:
            rows = list(qwen_windows(f))
        self.assertEqual(rows[0][1], 0)
        self.assertEqual(rows[-1][2], 123)
        for left, right in zip(rows, rows[1:]):
            self.assertEqual(left[2], right[1])
        self.assertTrue(all(0 < r[2] - r[1] <= 30 for r in rows))

    def model(self):
        path = self.root / "models/whisper-large-v3"
        path.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "weights.npz"):
            (path / name).write_text("fixture")
        atomic_json(path / "callrec-model.json", {"revision": MODEL_REVISION})

    def imported(self, seconds=1):
        session = create_session(self.root, "hindi", "import")
        target = session / "input.wav"
        wav(target, seconds)
        update_session(session, import_file="input.wav", import_sha256=sha256(target))
        return session

    def backend(self, samples, **kwargs):
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertIsNone(kwargs["language"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertTrue(kwargs["word_timestamps"])
        return {
            "language": "hi",
            "text": "आज English français",
            "segments": [
                {
                    "text": "आज English français",
                    "start": 0,
                    "end": 0.9,
                    "words": [
                        {"word": "आज", "start": 0.0, "end": 0.2},
                        {"word": " English", "start": 0.2, "end": 0.5},
                        {"word": " français", "start": 0.5, "end": 0.9},
                    ],
                }
            ],
        }

    def test_chunk_boundaries_and_recoverable_header(self):
        folder = self.root / "chunks"
        folder.mkdir()
        writer = ChunkWriter(folder, rate=10, channels=2, seconds=1)
        writer.write(bytes(range(40)) * 2 + b"\x00" * 20)
        # Even before close, wave header describes the written frames.
        with wave.open(str(folder / "000002.wav")) as f:
            self.assertEqual(f.getnframes(), 5)
        writer.close()
        self.assertEqual(writer.duration, 2.5)
        recovered = b""
        for p in sorted(folder.glob("*.wav")):
            with wave.open(str(p)) as f:
                recovered += f.readframes(f.getnframes())
        self.assertEqual(recovered, bytes(range(40)) * 2 + b"\x00" * 20)

    def test_disk_full_write_retains_previous_chunk(self):
        folder = self.root / "chunks"
        folder.mkdir()
        writer = ChunkWriter(folder, rate=10, channels=1, seconds=1)
        writer.write(b"\x01\x02" * 10)
        with patch("pathlib.Path.open", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                writer.write(b"\x00\x00")
        with wave.open(str(folder / "000000.wav")) as f:
            self.assertEqual(f.getnframes(), 10)

    def test_overlap_ownership_preserves_unicode_and_boundary(self):
        result = self.backend(
            None,
            task="transcribe",
            language=None,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        a = owned_segments(result, 23.8, 0, 24, 48)
        b = owned_segments(result, 23.8, 24, 48, 48)
        self.assertEqual(" ".join(x["text"] for x in a + b), "आज English français")
        self.assertEqual(len(list(windows(123 * 60))), 308)
        self.assertEqual(list(windows(24))[-1][2], 24)

    def test_unaligned_output_is_not_silently_dropped(self):
        with self.assertRaises(CallrecError):
            owned_segments({"text": "हिन्दी", "segments": [{"text": "हिन्दी"}]}, 0, 0, 24, 24)

    def test_end_to_end_import_offline_and_idempotent(self):
        self.model()
        session = self.imported()
        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(session, self.root, backend=self.backend)
        result = read_json(session / "transcript.json")
        self.assertTrue(result["processing_complete"])
        self.assertFalse(result["quality_verified"])
        self.assertIn("आज English français", (session / "transcript.md").read_text())
        self.assertEqual(result["processed_windows"], 1)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        before = sha256(session / "transcript.md")
        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(
                session,
                self.root,
                backend=lambda *a, **k: self.fail("retry re-transcribed completed audio"),
            )
        self.assertEqual(before, sha256(session / "transcript.md"))

    def test_failed_transcription_checkpoints_resume_without_duplicate(self):
        self.model()
        session = self.imported(seconds=50)
        calls = 0

        def failing(samples, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("model interrupted")
            return self.backend(samples, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                transcribe_session(session, self.root, backend=failing)
        self.assertEqual(read_json(session / "session.json")["status"], "transcription_failed")
        self.assertTrue((session / "checkpoints/000000.json").exists())
        self.assertIn("PARTIAL", (session / "transcript.partial.md").read_text())
        retries = 0

        def retry(samples, **kwargs):
            nonlocal retries
            retries += 1
            return {"language": "hi", "text": "", "segments": []}

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(session, self.root, backend=retry)
        self.assertEqual(retries, 2)
        self.assertEqual(len(read_json(session / "transcript.json")["segments"]), 1)
        self.assertEqual(read_json(session / "transcript.json")["audio_duration_seconds"], 50)

    def test_modified_source_rejects_checkpoint_reuse(self):
        session = self.imported()
        source = prepare_audio(session)
        with source.open("ab") as f:
            f.write(b"tamper")
        with self.assertRaises(CallrecError):
            prepare_audio(session)

    def test_lock_excludes_second_processing_operation(self):
        with lock(self.root / "lock"):
            with self.assertRaises(CallrecError):
                with lock(self.root / "lock"):
                    self.fail("lock was not exclusive")

    def test_stop_does_not_signal_unrelated_pid(self):
        session = self.imported()
        update_session(
            session, worker_pid=os.getpid(), capture_pid=os.getpid(), capture_command=["unrelated"]
        )
        with patch("callrec.cli.os.kill") as kill:
            stop_worker(session)
            kill.assert_not_called()

    def test_missing_permissions_fails_without_recording_claim(self):
        session = create_session(self.root, "general", "capture")
        fake = self.root / "fake-hark"
        fake.write_text("#!/bin/sh\necho 'permission denied' >&2\nexit 77\n")
        fake.chmod(0o755)
        # capture installs handlers, so restore them for unittest.
        signals = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        old = {sig: signal.getsignal(sig) for sig in signals}
        try:
            with patch("callrec.worker.hark_path", return_value=fake):
                capture(session, self.root)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        self.assertEqual(read_json(session / "session.json")["status"], "capture_failed")
        self.assertEqual(read_json(session / "session.json")["audio_seconds"], 0)
        self.assertIn("permission denied", (session / "capture.log").read_text())

    def test_cli_missing_input_has_no_session_side_effect(self):
        self.model()
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(
                ["--data-dir", str(self.root), "transcribe", str(self.root / "missing.wav")]
            )
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "sessions").exists())

    def install_fake_capture(self):
        folder = self.root / "tools" / f"hark-{HARK_VERSION}"
        folder.mkdir(parents=True)
        fake = folder / "hark"
        fake.write_text(f"""#!{sys.executable}
import os, signal, time
stop = False
def done(sig, frame):
    global stop
    stop = True
signal.signal(signal.SIGINT, done)
try:
    while not stop:
        os.write(1, b"\\x01\\x02" * 3200)
        time.sleep(0.1)
except BrokenPipeError:
    pass
""")
        fake.chmod(0o755)
        atomic_json(
            folder / "callrec-release.json",
            {"archive_sha256": HARK_SHA256, "binary_sha256": sha256(fake)},
        )

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "callrec.cli", "--data-dir", str(self.root), *args],
            capture_output=True,
            text=True,
            timeout=50,
        )

    def test_real_detached_lifecycle_and_second_start_refusal(self):
        self.install_fake_capture()
        try:
            first = self.run_cli("start", "--capture-only")
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            second = self.run_cli("start", "--capture-only")
            self.assertEqual(second.returncode, 1)
            self.assertIn("already exists", second.stderr)
            status = self.run_cli("status")
            self.assertEqual(status.returncode, 0, status.stdout)
        finally:
            stopped = self.run_cli("stop", "--audio-only")
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertFalse((self.root / "active.json").exists())
        session = next((self.root / "sessions").iterdir())
        self.assertTrue((session / "audio.m4a").is_file())

    def test_killed_worker_recovers_audio_and_stops_orphan(self):
        self.install_fake_capture()
        first = self.run_cli("start", "--capture-only")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        session = next((self.root / "sessions").iterdir())
        data = read_json(session / "session.json")
        os.kill(data["worker_pid"], signal.SIGKILL)
        time.sleep(0.1)
        stopped = self.run_cli("stop", "--audio-only")
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertTrue((session / "audio.m4a").is_file())
        self.assertIn("partial", read_json(session / "session.json")["capture_error"])

    def test_low_space_stops_native_capture_and_retains_audio(self):
        self.install_fake_capture()
        session = create_session(self.root, "general", "capture")
        signals = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        old = {sig: signal.getsignal(sig) for sig in signals}
        checks = 0

        def disk_check(path):
            nonlocal checks
            checks += 1
            if checks >= 3:
                raise CallrecError("Less than 256 MiB free")

        try:
            with patch("callrec.worker.check_space", side_effect=disk_check):
                capture(session, self.root)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        data = read_json(session / "session.json")
        self.assertEqual(data["status"], "capture_failed")
        self.assertIn("256 MiB", data["capture_error"])
        self.assertGreater(data["audio_seconds"], 0)
        self.assertTrue(data["capture_chunks"])


if __name__ == "__main__":
    unittest.main()
