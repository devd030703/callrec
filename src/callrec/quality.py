"""Conservative review signals. Findings never certify or rewrite speech."""

from __future__ import annotations

import collections
import re
import unicodedata

from .common import atomic_json, atomic_text, now, read_json, sha256


def findings(text, prompt=None):
    tokens = [
        "".join(
            c
            for c in unicodedata.normalize("NFC", word).casefold()
            if unicodedata.category(c)[0] in ("L", "M", "N")
        )
        for word in text.split()
    ]
    tokens = [w for w in tokens if w]
    reasons = []
    run = longest = 0
    previous = None
    for word in tokens:
        run = run + 1 if word == previous else 1
        longest = max(run, longest)
        previous = word
    if longest >= 8:
        reasons.append("consecutive_word_repetition")
    if len(tokens) >= 20 and max(collections.Counter(tokens).values()) / len(tokens) > 0.6:
        reasons.append("dominant_word_repetition")
    # Repeated phrases can loop without any one word dominating (observed in French).
    phrase_loop = False
    for width in range(2, min(12, len(tokens) // 4) + 1):
        for start in range(len(tokens) - 4 * width + 1):
            phrase = tokens[start : start + width]
            end = start + width
            while tokens[end : end + width] == phrase:
                end += width
            length = end - start
            if length >= max(16, 4 * width) and length / len(tokens) >= 0.6:
                phrase_loop = True
                break
        if phrase_loop:
            reasons.append("repeated_phrase_loop")
            break
    if "\ufffd" in text:
        reasons.append("unicode_replacement_character")

    def normalize(s):
        return re.sub(r"\s+", " ", s).strip().casefold()

    if prompt and normalize(prompt) in normalize(text):
        reasons.append("configured_prompt_leakage")
    elif prompt:
        for sentence in re.split(r"[.!?]\s*", prompt):
            if len(sentence.split()) >= 4 and normalize(sentence) in normalize(text):
                reasons.append("configured_prompt_fragment")
                break
    return reasons


def write_quality(session, checkpoints, settings, candidates=None):
    from .transcription import timestamp

    rows = []
    for c in checkpoints:
        text = " ".join(s["text"] for s in c["segments"])
        reasons = findings(text, settings.get("initial_prompt"))
        if c.get("decoding_recovery"):
            reasons.append("token_limit_recovery")
        if reasons:
            rows.append(
                {
                    "index": c["index"],
                    "start": c["core_start"],
                    "end": c["core_end"],
                    "reasons": reasons,
                    "original_text": text,
                }
            )
    previous = read_json(session / "quality.json", {})
    # Alternatives are tied to the exact checkpoints, never silently reused after changes.
    checkpoint_fingerprint = sha256_bytes(checkpoints)
    retained = (
        previous.get("alternatives", [])
        if previous.get("checkpoint_fingerprint") == checkpoint_fingerprint
        else []
    )
    report = {
        "schema_version": 1,
        "generated_at": now(),
        "quality_verified": False,
        "session_id": session.name,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "transcript_sha256_at_audit": sha256(session / "transcript.md")
        if (session / "transcript.md").exists()
        else None,
        "findings": rows,
        "alternatives": retained if candidates is None else candidates,
        "notice": "Review signals, not verified errors. Original text is retained. Alternatives are unverified.",
    }
    atomic_json(session / "quality.json", report)
    lines = ["# Transcript quality review", "", report["notice"], ""]
    for row in rows:
        lines += [
            f"## {timestamp(row['start'])}–{timestamp(row['end'])}",
            ", ".join(row["reasons"]),
            "",
            row["original_text"],
            "",
        ]
    for candidate in report["alternatives"]:
        lines += [
            f"### Alternative for window {candidate['index'] + 1}",
            f"Settings: {candidate['settings']}",
            "",
            candidate.get("text", candidate.get("error", "No candidate")),
            "",
        ]
    if not rows:
        lines += ["No configured signals detected. This does not verify accuracy."]
    atomic_text(session / "quality.md", "\n".join(lines) + "\n")
    return report


def sha256_bytes(value):
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
