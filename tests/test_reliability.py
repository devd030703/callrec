import contextlib
import io
import json
import os
import shutil
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

from callrec.cli import load_context, main, status
from callrec.common import (
    CallrecError,
    atomic_json,
    create_session,
    model_details,
    read_json,
    sha256,
    update_session,
)
from callrec.inference import ModelProcess
from callrec.jobs import enqueue, owned, run, snapshot
from callrec.quality import findings
from callrec.transcription import audit_session, transcribe_session


def fake_child(connection, engine, model, log_path):
    while True:
        try:
            request = connection.recv()
            if request is None:
                return
            samples, options = request
        except EOFError:
            return
        if model == "hang":
            time.sleep(20)
        elif model == "die":
            os._exit(2)
        connection.send(
            (True, {"text": "आज English français", "language": "Hindi", "generation_tokens": 5}, {})
        )


class Reliability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        details = model_details("qwen")
        folder = self.root / "models" / details["directory"]
        folder.mkdir(parents=True)
        for name in details["files"]:
            (folder / name).write_text("fixture")
        atomic_json(folder / "callrec-model.json", {"revision": details["revision"]})

    def tearDown(self):
        self.temp.cleanup()

    def session(self, seconds=1):
        s = create_session(self.root, "hindi", "import", "qwen")
        p = s / "input.wav"
        with wave.open(str(p), "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            w.writeframes(b"\0\0" * round(seconds * 16000))
        update_session(s, import_file=p.name, import_sha256=sha256(p))
        return s

    def complete(self, session, text="आज English français"):
        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(
                session,
                self.root,
                backend=lambda *a, **kw: SimpleNamespace(
                    text=text, language="Hindi", generation_tokens=20
                ),
            )

    def cached(self):
        s = self.session()
        self.complete(s)
        (s / "transcript.md").unlink()
        (s / "transcript.json").unlink()
        update_session(s, status="captured")
        return s

    def test_quality_real_loop_and_recovered_phrase_regressions(self):
        self.assertEqual(findings("अच्छा नमस्ते, हाँ, हाँ"), [])
        loop = "हाँ हाँ हाँ " + "नहीं " * 60
        self.assertIn("consecutive_word_repetition", findings(loop))
        self.assertIn("dominant_word_repetition", findings(loop))
        self.assertFalse(findings("नहीं नहीं नहीं, I made a mistake, je suis allé au marché"))
        self.assertIn("unicode_replacement_character", findings("गला�"))
        prompt = "Français et English. Aujourd’hui, nous étudions le français. French and English conversation."
        self.assertIn("configured_prompt_leakage", findings(prompt, prompt))
        self.assertFalse(findings("Français et English", prompt))
        self.assertIn(
            "configured_prompt_fragment", findings("French and English conversation.", prompt)
        )
        self.assertIn(
            "repeated_phrase_loop", findings("Les deux sont en train de se réunir. " * 12)
        )

    def test_suspect_alternatives_never_replace_original_and_are_bounded(self):
        s = self.session()
        text = "नहीं " * 60
        calls = []

        def model(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text=text if len(calls) == 1 else "यहाँ पर गलियाँ हैं",
                language="Hindi",
                generation_tokens=64,
            )

        transcribe_session(s, self.root, backend=model)
        self.assertEqual(len(calls), 3)
        self.assertIn(text.strip(), (s / "transcript.md").read_text())
        self.assertEqual(len(read_json(s / "quality.json")["alternatives"]), 2)
        self.assertIn("Review 00:00:00", (s / "transcript.md").read_text())
        originals = {
            name: sha256(s / name)
            for name in (
                "transcript.md",
                "transcript.json",
                "source.wav",
                "session.json",
                "checkpoints/000000.json",
            )
        }
        with contextlib.redirect_stdout(io.StringIO()):
            audit_session(s, self.root)
        self.assertEqual(originals, {name: sha256(s / name) for name in originals})

    def test_status_reports_latest_result_and_is_read_only(self):
        s = self.session()
        self.complete(s)
        original = {p.relative_to(s): sha256(p) for p in s.rglob("*") if p.is_file()}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(status(self.root, as_json=True), 0)
        self.assertEqual(json.loads(out.getvalue())["sessions"][0]["state"], "complete")
        self.assertEqual(
            original, {p.relative_to(s): sha256(p) for p in s.rglob("*") if p.is_file()}
        )

    def test_status_distinguishes_supervisor_and_model_memory(self):
        s = self.session()
        (s / "attempts").mkdir()
        atomic_json(
            s / "attempts" / "example.json",
            {
                "active_processing_seconds": 12.0,
                "model_process": {"peak_rss_mib": 1200.0},
            },
        )
        update_session(s, attempt_id="example", metrics={"process_peak_rss_mib": 30.0})
        metrics = snapshot(self.root, s)["metrics"]
        self.assertEqual(metrics["supervisor_peak_rss_mib"], 30.0)
        self.assertEqual(metrics["model_process_peak_rss_mib"], 1200.0)
        self.assertNotIn("process_peak_rss_mib", metrics)

    def test_context_is_bounded_and_recorded_in_checkpoint_settings(self):
        context_path = self.root / "context.json"
        atomic_json(context_path, {"hotwords": ["नवरात्रि", "analog clock"]})
        context = load_context(context_path, "qwen")
        s = self.session()
        atomic_json(s / "context.json", context)
        update_session(s, context_sha256=sha256(s / "context.json"))
        requests = []

        def backend(*args, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text="आज English", language="Hindi", generation_tokens=2)

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(s, self.root, backend=backend)
        report = read_json(s / "transcript.json")
        self.assertEqual(report["settings"]["hotwords"], context["hotwords"])
        self.assertEqual(requests[0]["hotwords"], context["hotwords"])
        atomic_json(context_path, {"hotwords": ["This is a complete sentence."]})
        with self.assertRaises(CallrecError):
            load_context(context_path, "qwen")
        with self.assertRaises(CallrecError):
            load_context(context_path, "whisper")

    def test_changed_context_refuses_resume(self):
        s = self.session()
        atomic_json(s / "context.json", {"hotwords": ["नमस्ते"]})
        update_session(s, context_sha256=sha256(s / "context.json"))
        atomic_json(s / "context.json", {"hotwords": ["दिवाली"]})
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(CallrecError, "context changed"):
                transcribe_session(s, self.root, backend=lambda *a, **kw: None)

    def test_larger_qwen_uses_its_own_revision_and_qwen_decoder(self):
        details = model_details("qwen-large")
        folder = self.root / "models" / details["directory"]
        folder.mkdir()
        for name in details["files"]:
            (folder / name).write_text("fixture")
        atomic_json(folder / "callrec-model.json", {"revision": details["revision"]})
        s = self.session()
        update_session(s, engine="qwen-large")
        seen = []

        def backend(*args, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(text="नमस्ते English", language="Hindi", generation_tokens=2)

        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(s, self.root, backend=backend)
        self.assertEqual(
            read_json(s / "transcript.json")["settings"]["model_revision"], details["revision"]
        )
        self.assertEqual(seen[0]["language"], "Hindi")

    def test_reprocess_copies_lossless_source_without_changing_parent(self):
        parent = self.session()
        shutil.copyfile(parent / "input.wav", parent / "source.wav")
        update_session(parent, status="complete", source_sha256=sha256(parent / "source.wav"))
        before = {p.relative_to(parent): sha256(p) for p in parent.rglob("*") if p.is_file()}
        with (
            patch("callrec.cli.enqueue", return_value="queued"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = main(
                ["--data-dir", str(self.root), "reprocess", parent.name, "--model", "qwen-0.6b"]
            )
        self.assertEqual(result, 0)
        child = next(p for p in (self.root / "sessions").iterdir() if p != parent)
        child_data = read_json(child / "session.json")
        self.assertEqual(child_data["parent_session_id"], parent.name)
        self.assertEqual(child_data["import_sha256"], sha256(parent / "source.wav"))
        self.assertEqual(
            before, {p.relative_to(parent): sha256(p) for p in parent.rglob("*") if p.is_file()}
        )

    def test_stale_pid_never_signalled_and_requires_explicit_retry(self):
        s = self.session()
        update_session(s, status="transcribing", processor_token="fake", processor_pid=os.getpid())
        atomic_json(self.root / "processor.json", {"pid": os.getpid(), "token": "fake"})
        with patch("callrec.jobs.os.kill", create=True) as kill:
            self.assertFalse(owned({"pid": os.getpid(), "token": "fake"}))
            item = snapshot(self.root, s)
            self.assertEqual(item["state"], "interrupted")
            self.assertEqual(item["retry_command"], f"callrec retry {s.name}")
            kill.assert_not_called()
        self.assertEqual(read_json(s / "session.json")["status"], "transcribing")

    def test_enqueue_idempotent_and_never_resurrects_abandoned_jobs(self):
        first, second = self.session(), self.session()
        update_session(first, status="queued", processor_token="dead")
        with (
            patch("callrec.jobs.owned", return_value=False),
            patch(
                "callrec.jobs.subprocess.Popen",
                return_value=SimpleNamespace(pid=12345, wait=lambda: 0),
            ),
        ):
            self.assertEqual(enqueue(self.root, second), "queued")
        self.assertEqual(read_json(first / "session.json")["status"], "interrupted")
        with (
            patch("callrec.jobs.owned", return_value=True),
            patch("callrec.jobs.subprocess.Popen") as spawn,
        ):
            self.assertIn("already", enqueue(self.root, second))
            spawn.assert_not_called()

    def test_queue_processes_creation_order_and_preserves_capture_pointer(self):
        first, second = self.cached(), self.cached()
        atomic_json(self.root / "processor.json", {"token": "test", "pid": os.getpid()})
        atomic_json(self.root / "active.json", {"id": second.name})
        for s in (second, first):
            update_session(s, status="queued", processor_token="test")
        order = []
        original = transcribe_session

        def record(s, root):
            order.append(s)
            return original(s, root)

        with patch("callrec.transcription.transcribe_session", side_effect=record):
            run(self.root, "test")
        self.assertEqual(order, [first, second])
        self.assertEqual(read_json(self.root / "active.json")["id"], second.name)
        self.assertFalse((self.root / "processor.json").exists())

    def test_detached_retry_survives_parent_exit_and_reuses_bytes(self):
        s = self.cached()
        before = sha256(s / "checkpoints/000000.json")
        # The launching shell exits immediately. Its detached supervisor must finish.
        command = [
            sys.executable,
            "-m",
            "callrec.cli",
            "--data-dir",
            str(self.root),
            "retry",
            s.name,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            d = read_json(s / "session.json")
            if d["status"] in ("complete", "complete_with_warnings", "transcription_failed"):
                break
            time.sleep(0.1)
        self.assertEqual(d["status"], "complete", d)
        self.assertEqual(before, sha256(s / "checkpoints/000000.json"))
        attempt = read_json(s / "attempts" / f"{d['attempt_id']}.json")
        self.assertEqual((attempt["reused_windows"], attempt["new_windows"]), (1, 0))

    def test_wait_interrupt_never_signals_supervisor(self):
        s = self.session()
        update_session(s, status="queued", processor_token="test")
        atomic_json(self.root / "processor.json", {"token": "test", "pid": 12345})
        with (
            patch("callrec.jobs.owned", return_value=True),
            patch("callrec.cli.time.sleep", side_effect=KeyboardInterrupt),
            patch("callrec.cli.os.kill") as kill,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(["--data-dir", str(self.root), "wait", s.name]), 130)
            kill.assert_not_called()
        self.assertEqual(read_json(s / "session.json")["status"], "queued")

    def test_model_child_deadline_restarts_once_and_does_not_kill_parent(self):
        events = []
        backend = ModelProcess(
            "qwen",
            "hang",
            self.root / "model.log",
            timeout=0.5,
            event=lambda kind, **kw: events.append((kind, kw)),
            target=fake_child,
        )
        try:
            with self.assertRaises(CallrecError):
                backend.window(lambda: backend([], language="Hindi"))
        finally:
            backend.close()
        self.assertEqual(sum(kind == "inference_started" for kind, _ in events), 2)
        self.assertEqual(sum(kind == "window_timeout" for kind, _ in events), 2)
        self.assertTrue(all(e[1]["pid"] != os.getpid() for e in events if "pid" in e[1]))

    def test_model_child_exit_is_visible(self):
        backend = ModelProcess("qwen", "die", self.root / "model.log", timeout=3, target=fake_child)
        try:
            with self.assertRaises(CallrecError):
                backend.window(lambda: backend([], language="Hindi"))
        finally:
            backend.close()

    def test_persistent_model_child_preserves_unicode(self):
        backend = ModelProcess("qwen", "ok", self.root / "model.log", timeout=3, target=fake_child)
        try:
            first = backend.window(lambda: backend([], language="Hindi"))
            pid = backend.process.pid
            second = backend.window(lambda: backend([], language="Hindi"))
            self.assertEqual(first.text, "आज English français")
            self.assertEqual(first.text, second.text)
            self.assertEqual(pid, backend.process.pid)
        finally:
            backend.close()

    def test_atomic_finalization_failure_is_retryable_without_reprocessing(self):
        s = self.session()
        from callrec.transcription import atomic_text as write

        def disk_failure(path, text):
            if path.name == "transcript.md":
                raise OSError(28, "No space left on device")
            return write(path, text)

        with (
            patch("callrec.transcription.atomic_text", side_effect=disk_failure),
            self.assertRaises(OSError),
        ):
            self.complete(s)
        self.assertEqual(read_json(s / "session.json")["status"], "transcription_failed")
        before = sha256(s / "checkpoints/000000.json")
        with contextlib.redirect_stdout(io.StringIO()):
            transcribe_session(s, self.root, backend=lambda *a, **k: self.fail("reprocessed audio"))
        self.assertEqual(before, sha256(s / "checkpoints/000000.json"))
        self.assertTrue(read_json(s / "transcript.json")["processing_complete"])
        self.assertEqual(read_json(s / "session.json")["status"], "complete")

    def test_stale_heartbeat_distinguished_from_absent_supervisor(self):
        s = self.session()
        update_session(
            s,
            status="transcribing",
            processor_token="test",
            processing_heartbeat_at="2000-01-01T00:00:00+00:00",
        )
        atomic_json(
            self.root / "processor.json",
            {"pid": 12345, "token": "test", "heartbeat_at": "2000-01-01T00:00:00+00:00"},
        )
        with patch("callrec.jobs.owned", return_value=True):
            self.assertIn("stalled", snapshot(self.root, s)["transcription_health"])
            self.assertEqual(snapshot(self.root, s)["state"], "transcribing")

    def test_stop_releases_capture_before_background_processing(self):
        from callrec.cli import finish

        s = self.session()
        atomic_json(self.root / "active.json", {"id": s.name})

        def queue(root, session):
            self.assertFalse((root / "active.json").exists())
            self.assertTrue((session / "audio.m4a").is_file())
            return "queued"

        with (
            patch("callrec.cli.enqueue", side_effect=queue),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            finish(self.root, s)
        self.assertEqual(read_json(s / "session.json")["status"], "captured")

    def test_terminated_supervisor_leaves_job_interrupted_until_explicit_retry(self):
        from callrec.common import lock

        s = self.cached()
        # Hold the model lock so the real detached supervisor stays alive but cannot finish.
        with lock(self.root / ".model.lock"):
            enqueue(self.root, s)
            owner = read_json(self.root / "processor.json")
            self.assertTrue(owned(owner))
            os.kill(owner["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 10
            while owned(owner) and time.monotonic() < deadline:
                time.sleep(0.1)
        self.assertFalse(owned(owner))
        item = snapshot(self.root, s)
        self.assertEqual(item["state"], "interrupted")
        self.assertTrue((s / "source.wav").is_file())

    def test_new_capture_works_while_processing_is_queued(self):
        from test_callrec import Tests

        from callrec.common import lock

        fixture = Tests(methodName="test_real_detached_lifecycle_and_second_start_refusal")
        fixture.root = self.root
        fixture.install_fake_capture()
        s = self.cached()
        with lock(self.root / ".model.lock"):
            enqueue(self.root, s)
            owner = read_json(self.root / "processor.json")
            try:
                started = fixture.run_cli("start", "--capture-only")
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                capture = read_json(self.root / "active.json")["id"]
                self.assertNotEqual(capture, s.name)
                self.assertEqual(read_json(s / "session.json")["status"], "queued")
            finally:
                stopped = fixture.run_cli("stop", "--audio-only")
                if owned(owner):
                    os.kill(owner["pid"], signal.SIGTERM)
            self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
            deadline = time.monotonic() + 10
            while owned(owner) and time.monotonic() < deadline:
                time.sleep(0.1)

    def test_empty_failed_capture_stop_unblocks_next_start_without_claiming_audio(self):
        from test_callrec import Tests

        fixture = Tests(methodName="test_real_detached_lifecycle_and_second_start_refusal")
        fixture.root = self.root
        fixture.install_fake_capture()
        s = create_session(self.root, "hindi", "capture", "qwen")
        update_session(
            s,
            status="capture_failed",
            audio_seconds=0,
            worker_pid=None,
            capture_error="No audio frames received",
        )
        atomic_json(self.root / "active.json", {"id": s.name})
        with patch("callrec.cli.enqueue") as queue, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.root), "stop"]), 0)
            queue.assert_not_called()
        self.assertFalse((self.root / "active.json").exists())
        self.assertFalse((s / "audio.m4a").exists())
        self.assertEqual(read_json(s / "session.json")["status"], "capture_failed")
        item = snapshot(self.root, s)
        self.assertEqual(item["capture_health"], "failed session closed; no audio captured")
        self.assertFalse(item["audio_saved"])
        try:
            result = fixture.run_cli("start", "--capture-only")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            fixture.run_cli("stop", "--audio-only")

    def test_startup_warmup_is_bounded_silent_output_and_not_a_recording_claim(self):
        from callrec.worker import start_silent_output_probe

        s = create_session(self.root, "hindi", "capture", "qwen")
        log = io.BytesIO()
        with patch("callrec.worker.subprocess.Popen") as spawn:
            start_silent_output_probe(s, log)
        with wave.open(str(s / ".startup-silence.wav")) as audio:
            self.assertEqual(audio.getnframes() / audio.getframerate(), 3)
            self.assertFalse(any(audio.readframes(audio.getnframes())))
        spawn.assert_called_once()
        self.assertEqual(spawn.call_args.args[0][0], "/usr/bin/afplay")
        data = read_json(s / "session.json")
        self.assertEqual(data["status"], "starting")
        self.assertEqual(data["capture_warmup_seconds"], 3)

    def test_idle_capture_gets_one_warmup_then_receives_real_pcm(self):
        from callrec.worker import capture

        s = create_session(self.root, "hindi", "capture", "qwen")
        ready = self.root / "output-awake"
        fake = self.root / "idle-hark"
        fake.write_text(
            f"#!{sys.executable}\nimport time, os\nfrom pathlib import Path\n"
            f"while not Path({str(ready)!r}).exists(): time.sleep(0.02)\n"
            "os.write(1, b'\\x01\\x02' * 16000)\n"
        )
        fake.chmod(0o755)

        def warmup(session, log):
            self.assertEqual(read_json(session / "session.json")["status"], "starting")
            ready.touch()

        signals = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        old = {sig: signal.getsignal(sig) for sig in signals}
        try:
            with (
                patch("callrec.worker.hark_path", return_value=fake),
                patch("callrec.worker.WARMUP_DELAY", 0.01),
                patch("callrec.worker.start_silent_output_probe", side_effect=warmup) as probe,
            ):
                capture(s, self.root)
                probe.assert_called_once()
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        self.assertGreater(read_json(s / "session.json")["audio_seconds"], 0)
        self.assertTrue(list((s / "chunks").glob("*.wav")))


if __name__ == "__main__":
    unittest.main()
