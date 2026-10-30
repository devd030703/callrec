# callrec

A small macOS CLI: record microphone + system audio, stop, then get full local
audio and timestamped original-language transcripts. No cloud transcription,
API keys, automatic summaries, vault writes, telemetry, or automatic deletion.

**Status: feasibility prototype.** Real lesson fidelity and both-source capture
must be checked before replacing your current recorder. A completed processing
job is not a verified transcript.

## Install once

Requires an Apple Silicon Mac, macOS 14.4+, Python 3.11–3.13, and ffmpeg.
Install Python 3.11 and ffmpeg with `brew install python@3.11 ffmpeg`.

```sh
git clone https://github.com/devd030703/callrec.git
cd callrec
./install.sh
callrec setup
callrec doctor
```

The installer copies the source into a versioned private application directory,
installs locked dependencies there and adds a launcher to `~/.local/bin`.
The checkout can subsequently be moved. The installer refuses to overwrite an
unrelated launcher or upgrade during active capture or processing. Older
application directories are retained. Add `~/.local/bin` to your PATH if needed.

`setup` is the explicit network step. It installs checksum-pinned Hark 0.4.3
and revision-pinned local models: Whisper large-v3 (~3 GB) plus Qwen3-ASR 0.6B
8-bit (~1 GB). Model setup uses small resumable HTTP byte ranges and verifies
pinned SHA-256 hashes before marking either model ready. Later commands
load a local model path with Hugging Face offline mode enabled. No subscriptions
are used. The upstream MLX Whisper package also brings PyTorch and other local
Python dependencies; no separate Python AI server is started.

## Before / after a call

```sh
callrec start --profile hindi   # or french; default profile: general
callrec status
callrec stop
```

`start` returns after actual PCM frames arrive. On first use macOS may request
Microphone and System Audio Recording permissions. Enable the listed Hark or
launching terminal in System Settings → Privacy & Security. Do not disable
Gatekeeper, reset privacy permissions, or grant unrelated permissions.

If capture has not delivered frames, `start` reports STARTING, returns exit code 2,
and the worker waits for up to 90 seconds. This does not establish that permission
was denied. After three seconds with no startup frames, callrec tries one bounded
three-second silent playback to wake an idle system-output clock. It plays no
sound, does not change devices or permission settings, and records the attempt
in capture.log and session metadata. Only received PCM can establish recording.
Run `status`; do not start another session.
`stop --audio-only` cancels/finalizes it. A permission grant or incoming frames
alone cannot prove both call participants are audible. Listen to your first
short recording using headphones and verify microphone AND remote sound.

`stop` finalizes and saves audio, queues transcription, and returns control to
the terminal. Transcription runs in a detached supervisor and survives closing
Terminal. `transcribe FILE` and `retry SESSION` also enqueue work. Only one model
worker runs at a time; sessions queue in creation order. You can record the next
call while an earlier session processes.

Use `callrec wait` to follow progress, or `callrec status --json` for structured
status. Ctrl-C in `wait` only stops waiting. `status` shows active capture,
processing/queued sessions and the latest result, even after capture stops.
`status SESSION` selects an older session. It does not change metadata.

The supervisor does not run at login. After logout, reboot, or a supervisor crash,
status reports interrupted work with an explicit `retry` command. Retrying one
session never silently restarts other abandoned jobs. Model inference runs in a
separate child with a 180-second complete-window deadline, including token-limit
retries and model loading. A timed-out window gets one retry in a fresh process.
A second timeout fails visibly; completed checkpoints and audio remain intact.
Native device/sleep behaviour still requires hardware checks.

Other useful commands:

```sh
callrec transcribe /absolute/path/to/recording.m4a --profile hindi
callrec stop --audio-only
callrec list
callrec wait
callrec status --json
callrec audit SESSION
callrec audit SESSION --retry-suspect
callrec retry 2026-10-24_12-00-00-example
callrec reprocess SESSION --model qwen-0.6b
```

Use `retry` with the actual ID from `list`, or its full session directory.
Existing recordings are copied into a fresh session; the original stays intact.
`reprocess` creates a linked session from the fingerprint-verified lossless
source; it leaves the earlier session unchanged. An already completed retry
does not overwrite its transcript. There is no
automatic source deduplication across separate imports.

## Optional vocabulary context

For Qwen, an explicitly supplied private JSON file can provide vocabulary hints:

```json
{"hotwords": ["नमस्ते", "analog clock"]}
```

Use `callrec start --profile hindi --context /path/to/words.json`, or pass
`--context` to `transcribe`. The file is copied into the private session and
fingerprinted so retries cannot silently change context. Hints can bias output or
be inserted without being spoken; their accuracy has not been validated. Avoid
full sentences, personal information and lesson answers. Context is never read
from a vault automatically. Original audio remains available for comparison.

## Local files and AI handoff

Default storage is **outside iCloud and the vault**:

```text
~/Library/Application Support/Callrec/sessions/<local-date-time-id>/
  audio.m4a                 Convenient AAC playback copy
  transcript.md             Full timestamped UTF-8 text
  transcript.json           Segments, timing resolution, settings and warnings
  session.json              State, provenance, durations, metrics, hashes
  chunks/000000.wav         Lossless 60-second mic + system mixes
  source.wav                Lossless normalized mono transcription source
  checkpoints/000000.json   Independently committed audio windows
  transcript.partial.md     Explicitly partial intermediate output
  capture.log               Native capture diagnostics
  worker.log                Capture supervisor diagnostics
  quality.md / quality.json Timestamped review signals and separate alternatives
  alternatives/             At most two alternatives per suspect Qwen window
  attempts/                 Per-attempt metadata, progress and model logs
  events.jsonl              Append-only job/recovery history
```

Point Claude/ChatGPT at `transcript.md`, or attach the file. Provide audio only
when you want that AI service to receive it. callrec does not invoke either
service. All files remain until you choose a retention policy. Partial transcripts
remain labelled partial even after a final transcript is available; use the final
`transcript.md` for intake.

Files are created privately (umask 077). There is no automatic encryption;
normal macOS account/FileVault protection applies. Root can be overridden before
the subcommand, for example `callrec --data-dir /local/path doctor`. Use the same
data directory for every command in a session.

## Language behaviour and limits

- Default routing is based on profile: **Hindi → local Qwen3-ASR 0.6B (8-bit)**;
  **French/general → local multilingual Whisper large-v3**. Both run through MLX.
  This changed after actual tests: Whisper auto romanized Hindi, while Qwen auto
  omitted Hindi or English passages. Neither automatic mode is a universal fix.
- Whisper uses `task=transcribe`, with language detection on each audio window.
  Qwen uses its ASR path with a Hindi decoding hint for the Hindi profile. The
  tested Hindi hint preserved English in Latin letters; it does not post-process
  English into Devanagari. No translation endpoint or transliteration is used.
- French provides short bilingual vocabulary context. Hindi selects the validated
  decoder/model combination. Intended result: Hindi in Devanagari,
  English in Latin letters, French accents preserved, including learner mistakes.
- Whisper: each 24-second core has up to 3 seconds of context on either side.
  Timestamped words are assigned by midpoint to a single core to avoid appending
  overlap twice. These are ASR estimates.
- Hindi/Qwen: contiguous chunks up to 12 seconds, cut at quiet points near 8
  seconds. No audio is removed. Timestamps honestly mark the **audio chunk**, not
  individual words. Qwen’s Hindi word alignment is not verified, so no precise
  word times are fabricated. Output that reaches its token limit gets at most two
  bounded 512-token retries with repetition penalties of 1.1 and 1.2. Recovered
  windows are identified in checkpoint metadata and final `recovery_events`, and
  the transcript carries a review warning. If both retries still exhaust their
  budgets, the job fails visibly without accepting truncated text. Normal
  decoding and existing checkpoint keys remain unchanged, so retries preserve
  earlier completed windows.
- Mixed-language/script performance is NOT guaranteed by selecting transcription.
  Whisper may still mistranscribe, change a script, omit speech or hallucinate.
  Use `--profile hindi` or `--profile french` for lessons; do not rely on general
  auto mode to preserve arbitrary mixed languages/scripts. A three-language
  Hindi/English/French auto benchmark failed. No LLM cleanup masks these errors.
  Missing Whisper word alignment fails the job rather
  than silently discarding nonempty speech at window boundaries.
- No speaker identification is inferred from the mixed audio; embedded exercises
  are not labelled as additional real participants.
- All system output is captured, including other apps' sounds. Default-microphone
  muting/routing can differ from the call app's own settings. Use recording only
  where participants and applicable policies permit it.

## Recovery and honest completeness

Each PCM buffer patches its WAV header; files are fsynced approximately once per
second and closed at 60 seconds. A normal stop drains the recorder before
assembly. Unexpected exit, no PCM frames for 20 seconds, or low free disk space
ends capture with a visible error and retains prior audio. No-frame startup has
a longer 90-second permission timeout.

Hark may restart streams after a device/sleep change, but that must be tested on
your actual devices. Wall-clock gaps are logged when input stops for more than
two seconds. Captured-audio timestamps do not reconstruct time or speech lost
during an interruption. Audio frames can also be silent: a healthy heartbeat
does not establish audible remote sound. Do not close the laptop mid-call.

`stop` recovers a dead capture worker and finalizes surviving chunks. It checks
process identity before sending signals. `retry` retains completed windows keyed
to source hash, model revision, profile and pipeline settings. It refuses changed
sources/checkpoints rather than mixing results. Final JSON and Markdown are each
atomic; session status is written last as their commit marker. Storage exhaustion
can prevent even error metadata from being written; audio already on disk remains.

## Transcript review and timing

Every window is checked for eight consecutive identical normalized words, one
word occupying more than 60% of at least twenty words, Unicode replacement
characters, and the exact configured prompt (or its sentences of at least four words). A repeated phrase also triggers
review when a 2–12-word phrase repeats at least four times, covering at least
sixteen words and 60% of the window. These checks are review signals, not verified
errors: legitimate repetition and learner mistakes stay in the original text.

For new Qwen jobs, suspect windows receive at most two separate alternative
candidates, with repetition penalties 1.1 and 1.2. Candidates and failures are
reserved/persisted before inference so a retry cannot keep generating more.
Each candidate has the same timeout/restart protection. Original completed text
is never replaced. Token-exhausted windows have no accepted original output;
the existing bounded token-recovery policy still supplies explicitly flagged text.
Whisper findings are flagged without automatic second-model rewriting.

`audit SESSION` adds `quality.md` and `quality.json` to an existing complete
session. `--retry-suspect` also generates local Qwen alternatives; it refuses to
compete with a busy model worker. Audits never alter existing transcripts,
audio, session metadata or checkpoints. Read `quality.md` alongside the audio.
No finding-free report establishes that every spoken word was transcribed.

Attempt metadata records new/reused windows, settings, timestamps, active elapsed
processing time and model-process peak RSS. Session metadata accumulates only
recorded attempt durations; older metrics are explicitly incomplete. Parent and
model RSS are not a complete accounting of unified GPU memory. Model children
use offline flags and reject Python IPv4/IPv6 connections; no models are fetched
by recording, transcription or audit.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
# Optional actual-model checks using generated Mac voices; models must be installed:
CALLREC_MODEL_SMOKE=1 .venv/bin/python -m unittest discover -s tests -v
```

The standard suite uses synthetic PCM and mocked ASR for failure, overlap,
locking, checkpoints and provenance. The opt-in suite generates Hindi, French
and English speech, denies Python socket connections, and checks native script,
accents, repeated closing sentences and multiple windows with the real models.
Neither suite verifies real lesson fidelity or both-source macOS permissions.
These tests use generated speech and do not establish real-lesson accuracy.

Before using this as your sole recorder, verify a short bilingual Hindi clip,
a French clip, and then one full lesson of each language, including the final
sentence. The selected Hindi Qwen path was adopted only after comparison on the same
synthetic Hindi/English recording. The unhinted Qwen path and forced-French
decoding omitted speech, so they are not the defaults. Experimental engine
overrides (`--engine qwen|whisper`) are available for explicit comparisons; do
not use them as a quality guarantee. `setup --engine whisper|qwen` can install
one model instead of both.

## Upstream pins

- Hark 0.4.3: <https://github.com/PhantomYdn/hark/releases/tag/v0.4.3>
- Archive SHA-256: `5301df867b44f68f686a4b9230bfe7b9d9233f0091054fde1ac7560639e7fcae`
- Model: <https://huggingface.co/mlx-community/whisper-large-v3-mlx>
- Model revision: `49e6aa286ad60c14352c404340ded53710378a11`
- MLX Whisper: `0.4.3`; exact resolved Python dependencies in `uv.lock`.

Hark's upstream licenses/notices are retained alongside the installed binary.

- Hindi model: <https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit>
- Hindi model revision: `89e96d92ba34aca20b3e29fb10cc284097d1219f`; every file size/hash is pinned in `src/callrec/qwen-manifest.json`.
- MLX Audio: `0.6.1`.

## Benchmarking known references

`python benchmarks/score.py results.json` compares baseline and candidate text
against supplied reference text. The input is a JSON array; each entry contains
`id`, `language`, `reference`, `baseline`, and `candidate`, with optional
`required_phrases` and `forbidden_phrases` lists. The report preserves script and
accent distinctions, reports word and character error rates, and applies the
promotion gate. Only use independently established reference text for scoring.
Class notes and model outputs do not establish what was spoken. A passing score
is a benchmark result, not verification of an individual transcript.
