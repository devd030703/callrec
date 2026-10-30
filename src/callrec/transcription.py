from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
import wave

from .common import (
    PROFILES,
    CallrecError,
    atomic_json,
    atomic_text,
    duration,
    lock,
    model_details,
    model_path,
    now,
    read_json,
    run,
    sha256,
    tool,
    update_session,
)

PIPELINE_VERSION = 3
CORE_SECONDS = 24.0
CONTEXT_SECONDS = 3.0


def prepare_audio(session):
    data = read_json(session / "session.json")
    source = session / "source.wav"
    if data.get("source_sha256") and source.is_file():
        if sha256(source) != data["source_sha256"]:
            raise CallrecError(
                "Normalized audio changed since checkpointing; refusing to reuse transcript checkpoints."
            )
        return source
    chunks = sorted((session / "chunks").glob("*.wav"))
    if data["kind"] == "capture":
        if not chunks:
            raise CallrecError(
                "No captured audio exists. Check capture.log and macOS recording permissions."
            )
        # WAV headers are patched on every write, including the current partial chunk.
        total = sum(duration(p) for p in chunks)
        atomic_text(session / "concat.txt", "".join(f"file 'chunks/{p.name}'\n" for p in chunks))
        inputs = ["-f", "concat", "-safe", "1", "-i", str(session / "concat.txt")]
    else:
        original = session / data["import_file"]
        if sha256(original) != data["import_sha256"]:
            raise CallrecError("Imported audio changed; refusing to reuse checkpoints.")
        total = duration(original)
        inputs = ["-i", str(original)]
    tmp = session / ".source.partial.wav"
    run(
        [
            tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-y",
            *inputs,
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            tmp,
        ]
    )
    measured = duration(tmp)
    if abs(total - measured) > max(0.15, total * 0.001):
        raise CallrecError(
            f"Audio assembly duration mismatch: expected {total:.3f}s, got {measured:.3f}s."
        )
    os.replace(tmp, source)
    update_session(
        session,
        source_sha256=sha256(source),
        source_duration_seconds=measured,
        capture_chunks=[
            {"file": p.name, "sha256": sha256(p), "duration_seconds": duration(p)} for p in chunks
        ],
    )
    return source


def playback_audio(session, source):
    target = session / "audio.m4a"
    if not target.exists():
        tmp = session / ".audio.partial.m4a"
        run(
            [
                tool("ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                source,
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                tmp,
            ]
        )
        if abs(duration(tmp) - duration(source)) > 0.2:
            raise CallrecError("Playback audio duration mismatch; lossless source retained.")
        os.replace(tmp, target)
    return target


def windows(total):
    for i in range(math.ceil(total / CORE_SECONDS)):
        core_start = i * CORE_SECONDS
        core_end = min(total, core_start + CORE_SECONDS)
        yield (
            i,
            core_start,
            core_end,
            max(0.0, core_start - CONTEXT_SECONDS),
            min(total, core_end + CONTEXT_SECONDS),
        )


def qwen_windows(audio):
    """Cover every sample once, choosing quiet cut points near 8 seconds.

    Qwen has no verified Hindi word aligner. Report honest audio-chunk times
    instead of fabricated word times or unsafe overlap-text deduplication.
    """
    import numpy as np

    rate = audio.getframerate()
    frames = audio.getnframes()
    start = 0
    index = 0
    while start < frames:
        if frames - start <= 12 * rate:
            end = frames
        else:
            lo, hi = start + 6 * rate, start + 10 * rate
            audio.setpos(lo)
            samples = np.frombuffer(audio.readframes(hi - lo), dtype=np.int16).astype(np.float64)
            # Mean energy over 100ms; no speech/non-speech frames are removed.
            width = rate // 10
            energies = np.mean(
                samples[: len(samples) // width * width].reshape(-1, width) ** 2, axis=1
            )
            candidates = np.flatnonzero(energies <= energies.min() + 1e-10)
            desired = 2 * rate / width - 0.5
            chosen = candidates[np.argmin(abs(candidates - desired))]
            end = lo + int((chosen + 0.5) * width)
        yield index, start / rate, end / rate, start / rate, end / rate
        start = end
        index += 1


def owned_segments(result, start, core_start, core_end, total):
    """Assign overlap words to a single core interval using audio timestamps."""
    segments = []
    for segment in result.get("segments", []):
        if not str(segment.get("text", "")).strip():
            continue
        if segment.get("words"):
            words = []
            for word in segment["words"]:
                ws, we = start + float(word["start"]), start + float(word["end"])
                if not math.isfinite(ws + we) or we < ws:
                    raise CallrecError("Model returned invalid word timestamps.")
                middle = (ws + we) / 2
                if core_start <= middle < core_end or (core_end == total and middle == total):
                    words.append(
                        {"word": word["word"], "start": max(0.0, ws), "end": min(total, we)}
                    )
            if not words:
                continue
            text = "".join(w["word"] for w in words).strip()
            begin, end = words[0]["start"], words[-1]["end"]
        else:
            # Never silently discard text when timestamp alignment fails.
            # Overlap text cannot safely be assigned without word timestamps.
            raise CallrecError(
                "Nonempty ASR text lacks word alignment. Audio retained; retry rather than silently omit boundary speech."
            )
        segments.append(
            {
                "start": round(begin, 3),
                "end": round(end, 3),
                "text": text,
                "language_detected": result.get("language"),
                "words": words,
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
            }
        )
    if result.get("text", "").strip() and not result.get("segments"):
        raise CallrecError("ASR returned text without segments; refusing an incomplete transcript.")
    return segments


def timestamp(value):
    milliseconds = int(round(value * 1000))
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def write_partial(session, checkpoints, expected):
    segments = [s for c in checkpoints for s in c["segments"]]
    lines = [
        "# PARTIAL — transcription has not finished",
        "",
        f"Processed {len(checkpoints)}/{expected} audio windows. Retry this session to resume.",
        "",
    ]
    lines += [f"[{timestamp(s['start'])} – {timestamp(s['end'])}] {s['text']}" for s in segments]
    atomic_text(session / "transcript.partial.md", "\n".join(lines) + "\n")


def decode_qwen(backend, samples, language, prompt, hotwords=None):
    """Retry only token-exhausted windows, without accepting truncated output.

    Normal decoding and existing checkpoint keys remain unchanged. Each
    recovered window records its different settings for review and resume.
    """
    options = {
        "language": language,
        "max_tokens": 2048,
        "temperature": 0.0,
        "system_prompt": prompt,
        "verbose": False,
    }
    if hotwords:
        options["hotwords"] = hotwords
    output = backend(samples, **options)
    if output.generation_tokens < options["max_tokens"]:
        return output, None
    recovery = {
        "policy": "qwen-token-recovery-v1",
        "trigger": "output_token_limit",
        "original_generation_tokens": output.generation_tokens,
        "attempts": [],
    }
    for penalty in (1.1, 1.2):
        print(
            f"Token limit reached; retrying this audio window (repetition penalty {penalty})…",
            flush=True,
        )
        output = backend(
            samples,
            **{**options, "max_tokens": 512},
            repetition_penalty=penalty,
            repetition_context_size=128,
        )
        recovery["attempts"].append(
            {
                "max_tokens": 512,
                "repetition_penalty": penalty,
                "repetition_context_size": 128,
                "generation_tokens": output.generation_tokens,
            }
        )
        if output.generation_tokens < 512:
            return output, recovery
    raise CallrecError(
        "Qwen hit its output token limit after two bounded recovery attempts; "
        "audio and completed checkpoints retained. This window is not marked complete."
    )


def _transcribe_session(session, root, backend=None, attempt_id=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    metadata = read_json(session / "session.json")
    if metadata["status"] in ("complete", "complete_with_warnings"):
        final = read_json(session / "transcript.json", {})
        if final.get("processing_complete") and (session / "transcript.md").is_file():
            print(f"Already transcribed: {session / 'transcript.md'}")
            return
        raise CallrecError(
            "Completed session has missing output. Original audio is retained; investigate before overwriting."
        )
    started = time.monotonic()
    update_session(
        session, status="transcribing", transcription_started_at=now(), transcription_error=None
    )
    try:
        source = prepare_audio(session)
        playback_audio(session, source)
        engine = metadata.get("engine", "whisper")
        details = model_details(engine)
        model = model_path(root, engine)
        config = {
            "pipeline_version": PIPELINE_VERSION,
            "engine": engine,
            "model_repo": details["repo"],
            "model_revision": details["revision"],
            "source_sha256": sha256(source),
            "profile": metadata["profile"],
            "task": "transcribe",
            "language": {"hindi": "Hindi", "french": "French", "general": None}[metadata["profile"]]
            if engine in ("qwen", "qwen-large")
            else None,
            "condition_on_previous_text": False,
            "word_timestamps": engine == "whisper",
            "timestamp_resolution": "word" if engine == "whisper" else "audio_chunk",
            "core_seconds": CORE_SECONDS if engine == "whisper" else 8,
            "context_seconds": CONTEXT_SECONDS if engine == "whisper" else 0,
            "initial_prompt": PROFILES[metadata["profile"]] if engine == "whisper" else None,
        }
        if metadata.get("context_sha256"):
            context_path = session / "context.json"
            if not context_path.is_file() or sha256(context_path) != metadata["context_sha256"]:
                raise CallrecError("Saved vocabulary context changed; refusing to mix checkpoints.")
            config["context_sha256"] = metadata["context_sha256"]
            config["hotwords"] = read_json(context_path)["hotwords"]
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        update_session(session, processing_settings=config)
        import numpy as np

        from .jobs import event
        from .quality import findings, write_quality

        with wave.open(str(source), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16000
            ):
                raise CallrecError("Unexpected normalized audio format.")
            total = audio.getnframes() / 16000
            work = list(windows(total) if engine == "whisper" else qwen_windows(audio))
            update_session(session, expected_windows=len(work))
            checkpoints = []
            reused_windows = new_windows = 0
            for index, core_start, core_end, begin, end in work:
                checkpoint = session / "checkpoints" / f"{index:06d}.json"
                cached = read_json(checkpoint)
                if cached is not None:
                    if cached.get("key") != key or cached.get("index") != index:
                        raise CallrecError(
                            "Checkpoint input/settings mismatch. Create a new session rather than mixing transcripts."
                        )
                    result = cached
                    reused_windows += 1
                    event(session, "window_reused", attempt_id=attempt_id, index=index)
                else:
                    recovery = None
                    print(
                        f"Transcribing {index + 1}/{len(work)} [{timestamp(core_start)}–{timestamp(core_end)}]",
                        flush=True,
                    )
                    audio.setpos(round(begin * 16000))
                    samples = (
                        np.frombuffer(
                            audio.readframes(round((end - begin) * 16000)), dtype=np.int16
                        ).astype(np.float32)
                        / 32768.0
                    )
                    update_session(
                        session,
                        current_window={"index": index, "start": core_start, "end": core_end},
                        window_started_at=now(),
                    )

                    def infer():
                        if engine in ("qwen", "qwen-large"):
                            return decode_qwen(
                                backend,
                                samples,
                                config["language"],
                                config["initial_prompt"],
                                config.get("hotwords"),
                            )
                        return backend(
                            samples,
                            path_or_hf_repo=str(model),
                            task="transcribe",
                            language=None,
                            initial_prompt=config["initial_prompt"],
                            condition_on_previous_text=False,
                            word_timestamps=True,
                            verbose=None,
                            temperature=0.0,
                        ), None

                    output, recovery = (
                        backend.window(infer) if hasattr(backend, "window") else infer()
                    )
                    if engine == "whisper":
                        language = output.get("language")
                        segments = owned_segments(output, begin, core_start, core_end, total)
                    else:
                        language = output.language
                        text = output.text.strip()
                        segments = (
                            [
                                {
                                    "start": core_start,
                                    "end": core_end,
                                    "text": text,
                                    "language_detected": language,
                                    "timestamp_resolution": "audio_chunk",
                                }
                            ]
                            if text
                            else []
                        )
                    result = {
                        "key": key,
                        "index": index,
                        "core_start": core_start,
                        "core_end": core_end,
                        "language_detected": language,
                        "segments": segments,
                    }
                    if recovery is not None:
                        result["decoding_recovery"] = recovery
                    atomic_json(checkpoint, result)
                    new_windows += 1
                    event(
                        session,
                        "window_completed",
                        attempt_id=attempt_id,
                        index=index,
                        recovered=recovery is not None,
                    )
                update_session(
                    session, attempt_reused_windows=reused_windows, attempt_new_windows=new_windows
                )
                if engine in ("qwen", "qwen-large") and findings(
                    " ".join(s["text"] for s in result["segments"])
                ):
                    retry_candidates(session, result, config, backend, audio)
                checkpoints.append(result)
                write_partial(session, checkpoints, len(work))
                update_session(
                    session,
                    processed_windows=len(checkpoints),
                    expected_windows=len(work),
                    processed_through_seconds=core_end,
                )
        segments = [s for c in checkpoints for s in c["segments"]]
        segments.sort(key=lambda x: x["start"])
        warnings = [metadata["warning"]]
        recovery_events = [
            {
                "index": c["index"],
                "start": c["core_start"],
                "end": c["core_end"],
                **c["decoding_recovery"],
            }
            for c in checkpoints
            if c.get("decoding_recovery")
        ]
        if recovery_events:
            warnings.append(
                f"{len(recovery_events)} audio window(s) recovered from Qwen token limits "
                "using bounded retries with repetition penalties. Review these regions; "
                "details are in transcript.json recovery_events."
            )
        current = read_json(session / "session.json")
        if current.get("startup_delay_seconds", 0) > 2:
            warnings.append(
                f"No audio was received during the first {current['startup_delay_seconds']:.1f}s "
                "after capture started; that interval is missing from the recording."
            )
        if current.get("capture_error"):
            warnings.append(current["capture_error"])
        if current.get("interruptions"):
            warnings.append(
                "Capture interruptions occurred. Timestamps index captured audio, not missing wall-clock time."
            )
        if not segments:
            warnings.append(
                "No speech was recognized. This does not verify that the recording was silent."
            )
        quality = write_quality(
            session, checkpoints, config, candidates=all_candidates(session, checkpoints)
        )
        for finding in quality["findings"]:
            warnings.append(
                f"Review {timestamp(finding['start'])}–{timestamp(finding['end'])}: "
                f"{', '.join(finding['reasons'])}. Original text retained; see quality.md."
            )
        seconds = round(time.monotonic() - started, 3)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, unlike Linux's KiB.
        import sys

        peak_mib = round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
        report = {
            "schema_version": 2,
            "processing_complete": True,
            "quality_verified": False,
            "session_id": metadata["id"],
            "settings": config,
            "audio_duration_seconds": total,
            "processed_windows": len(checkpoints),
            "expected_windows": len(work),
            "warnings": warnings,
            "recovery_events": recovery_events,
            "quality_findings": quality["findings"],
            "segments": segments,
            "metrics": {"transcription_wall_seconds": seconds, "process_peak_rss_mib": peak_mib},
        }
        lines = [
            "# Call transcript",
            "",
            f"Session: {metadata['id']} · Profile: {metadata['profile']}",
            "",
            f"Audio: {timestamp(total)} · All {len(work)} audio windows processed.",
            "",
            "> Machine transcript; not human-verified. Original-language transcription; no speaker identities inferred.",
            "",
        ]
        lines += [f"> {w}" for w in warnings[1:]]
        if engine in ("qwen", "qwen-large"):
            lines += ["> Timestamps mark audio chunks, not exact word or sentence boundaries."]
        lines += ["", "## Transcript", ""]
        lines += [
            f"[{timestamp(s['start'])} – {timestamp(s['end'])}] {s['text']}" for s in segments
        ]
        # Session status is the commit marker after both independently atomic outputs.
        atomic_json(session / "transcript.json", report)
        atomic_text(session / "transcript.md", "\n".join(lines) + "\n")
        update_session(
            session,
            pending_completion_status="complete_with_warnings" if len(warnings) > 1 else "complete",
            transcription_finished_at=now(),
            metrics=report["metrics"],
            warnings=warnings,
            transcript_sha256=sha256(session / "transcript.md"),
            current_window=None,
        )
        print(
            f"Audio: {session / 'audio.m4a'}\nTranscript: {session / 'transcript.md'}\nJSON: {session / 'transcript.json'}"
        )
        print(
            f"Measured processing: {seconds:.1f}s; process peak RSS: {peak_mib:.1f} MiB. Fidelity unverified."
        )
    except BaseException as e:
        update_session(
            session, status="transcription_failed", transcription_error=f"{type(e).__name__}: {e}"
        )
        raise


def retry_candidates(session, checkpoint, config, backend, audio):
    """At most two alternatives per exact checkpoint; never replace original text."""
    from .quality import findings, sha256_bytes

    folder = session / "alternatives"
    folder.mkdir(exist_ok=True)
    path = folder / f"{checkpoint['index']:06d}.json"
    key = sha256_bytes({"checkpoint": checkpoint, "policy": "quality-alternatives-v1"})
    stored = read_json(path, {})
    if stored and stored.get("key") != key:
        raise CallrecError("Alternative checkpoint changed; refusing to mix review candidates.")
    candidates = stored.get("candidates", [])
    import numpy as np

    audio.setpos(round(checkpoint["core_start"] * 16000))
    samples = (
        np.frombuffer(
            audio.readframes(round((checkpoint["core_end"] - checkpoint["core_start"]) * 16000)),
            dtype=np.int16,
        ).astype(np.float32)
        / 32768.0
    )
    for penalty in (1.1, 1.2)[len(candidates) :]:
        candidate = {
            "index": checkpoint["index"],
            "start": checkpoint["core_start"],
            "end": checkpoint["core_end"],
            "settings": {
                "repetition_penalty": penalty,
                "repetition_context_size": 128,
                "max_tokens": 512,
                "temperature": 0.0,
                "language": config["language"],
            },
            "quality_verified": False,
        }
        if config.get("hotwords"):
            candidate["settings"]["hotwords"] = config["hotwords"]
        # Reserve the attempt before inference, so a killed process cannot exceed the cap.
        candidate["error"] = "Alternative interrupted before completion; not automatically retried."
        candidates.append(candidate)
        atomic_json(path, {"key": key, "candidates": candidates})
        try:

            def operation():
                return backend(samples, system_prompt=None, verbose=False, **candidate["settings"])

            output = backend.window(operation) if hasattr(backend, "window") else operation()
            if output.generation_tokens >= 512:
                raise CallrecError("Alternative exhausted token budget; not accepted as complete.")
            candidate.pop("error", None)
            candidate.update(
                text=output.text.strip(),
                generation_tokens=output.generation_tokens,
                findings=findings(output.text),
            )
        except Exception as e:
            candidate["error"] = f"{type(e).__name__}: {e}"
        atomic_json(path, {"key": key, "candidates": candidates})
    return candidates


def all_candidates(session, checkpoints):
    from .quality import sha256_bytes

    result = []
    for checkpoint in checkpoints:
        stored = read_json(session / "alternatives" / f"{checkpoint['index']:06d}.json", {})
        key = sha256_bytes({"checkpoint": checkpoint, "policy": "quality-alternatives-v1"})
        if stored.get("key") == key:
            result.extend(stored["candidates"])
    return result


def transcribe_session(session, root, backend=None):
    import contextlib
    import traceback
    import uuid

    from . import __version__
    from .inference import ModelProcess
    from .jobs import event

    metadata = read_json(session / "session.json")
    if metadata["status"] in ("complete", "complete_with_warnings"):
        final = read_json(session / "transcript.json", {})
        if final.get("processing_complete") and (session / "transcript.md").is_file():
            print(f"Already transcribed: {session / 'transcript.md'}")
            return
        raise CallrecError("Completed session has missing output; refusing to overwrite.")
    attempt_id = uuid.uuid4().hex
    folder = session / "attempts"
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    attempt = {
        "id": attempt_id,
        "started_at": now(),
        "version": __version__,
        "engine": metadata.get("engine", "whisper"),
        "status": "running",
        "reused_windows": 0,
        "new_windows": 0,
    }
    atomic_json(folder / f"{attempt_id}.json", attempt)
    fields = {
        "attempts": metadata.get("attempts", []) + [attempt_id],
        "attempt_id": attempt_id,
        "attempt_reused_windows": 0,
        "attempt_new_windows": 0,
    }
    if metadata.get("metrics") and not metadata.get("attempts"):
        fields["legacy_metrics"] = {"incomplete": True, "reported": metadata["metrics"]}
    update_session(session, **fields)
    event(
        session,
        "attempt_started",
        attempt_id=attempt_id,
        version=__version__,
        engine=attempt["engine"],
    )

    def inference_event(kind, **fields):
        event(session, kind, attempt_id=attempt_id, **fields)
        if kind == "inference_started":
            update_session(session, inference_pid=fields["pid"])
        elif kind == "inference_stopped":
            update_session(session, inference_pid=None)
        elif kind == "inference_window_started":
            update_session(session, window_started_at=now())

    model_backend = None
    try:
        if backend is None:
            model_backend = ModelProcess(
                attempt["engine"],
                model_path(root, attempt["engine"]),
                folder / f"{attempt_id}.model.log",
                event=inference_event,
            )
            backend = model_backend
        with (folder / f"{attempt_id}.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    _transcribe_session(session, root, backend, attempt_id)
                except BaseException:
                    traceback.print_exc()
                    raise
        attempt["status"] = "complete"
    except BaseException as e:
        attempt.update(status="failed", error=f"{type(e).__name__}: {e}")
        update_session(session, status="transcription_failed", transcription_error=attempt["error"])
        raise
    finally:
        if model_backend:
            model_backend.close()
        data = read_json(session / "session.json")
        attempt.update(
            finished_at=now(),
            active_processing_seconds=round(time.monotonic() - started, 3),
            reused_windows=data.get("attempt_reused_windows", 0),
            new_windows=data.get("attempt_new_windows", 0),
        )
        report = read_json(session / "transcript.json", {})
        attempt["settings"] = data.get("processing_settings", report.get("settings"))
        if model_backend:
            attempt["model_process"] = model_backend.stats
        atomic_json(folder / f"{attempt_id}.json", attempt)
        update_session(
            session,
            cumulative_processing_seconds=round(
                data.get("cumulative_processing_seconds", 0) + attempt["active_processing_seconds"],
                3,
            ),
            current_window=None,
        )
        event(
            session,
            "attempt_finished",
            attempt_id=attempt_id,
            **{
                k: attempt[k]
                for k in ("status", "active_processing_seconds", "reused_windows", "new_windows")
            },
        )
        if attempt["status"] == "complete":
            update_session(
                session, status=data["pending_completion_status"], pending_completion_status=None
            )


def audit_session(session, root, retry_suspect=False):
    from .inference import ModelProcess
    from .jobs import event
    from .quality import findings, write_quality

    # Audit is additive, including for older completed sessions.
    with lock(session / ".process.lock"):
        report = read_json(session / "transcript.json", {})
        if not report.get("processing_complete"):
            raise CallrecError(
                "Audit requires a completed transcript. Retry incomplete transcription first."
            )
        checkpoints = [read_json(p) for p in sorted((session / "checkpoints").glob("*.json"))]
        if len(checkpoints) != report["expected_windows"]:
            raise CallrecError(
                "Checkpoint coverage differs from transcript; investigate before audit."
            )
        settings = report["settings"]
        write_quality(session, checkpoints, settings, all_candidates(session, checkpoints))
        if retry_suspect and settings["engine"] in ("qwen", "qwen-large"):
            with lock(root / ".model.lock"):
                source = prepare_audio(session)
                backend = ModelProcess(
                    settings["engine"],
                    model_path(root, settings["engine"]),
                    session / "audit.model.log",
                    event=lambda kind, **fields: event(session, kind, operation="audit", **fields),
                )
                try:
                    with wave.open(str(source), "rb") as audio:
                        for checkpoint in checkpoints:
                            text = " ".join(s["text"] for s in checkpoint["segments"])
                            if findings(text):
                                retry_candidates(session, checkpoint, settings, backend, audio)
                finally:
                    backend.close()
        quality = write_quality(
            session, checkpoints, settings, all_candidates(session, checkpoints)
        )
        event(
            session,
            "audit_completed",
            findings=len(quality["findings"]),
            alternatives=len(quality["alternatives"]),
        )
        print(
            f"Review signals: {len(quality['findings'])}; alternatives: {len(quality['alternatives'])}\nReview: {session / 'quality.md'}"
        )
        return quality
