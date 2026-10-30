#!/bin/zsh
set -eu
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$HOME/Library/Application Support/Callrec"
PYTHON_BIN="${CALLREC_PYTHON:-$(command -v python3.11 || true)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 'Python 3.11 is required. Install it with brew install python@3.11, or set CALLREC_PYTHON.'
  exit 1
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  print -u2 'Install ffmpeg first: brew install ffmpeg'
  exit 1
fi
umask 077
mkdir -p "$APP_DIR/app/releases" "$HOME/.local/bin"
"$PYTHON_BIN" - "$APP_DIR" "$HOME/.local/bin/callrec" <<'PY'
import json, pathlib, subprocess, sys
root, launcher = map(pathlib.Path, sys.argv[1:])
if launcher.exists() or launcher.is_symlink():
    if launcher.is_symlink() or 'callrec-managed-launcher' not in launcher.read_text():
        raise SystemExit(f'Refusing to overwrite unrelated launcher: {launcher}')
    result = subprocess.run([str(launcher), 'status', '--json'], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit('Cannot inspect the existing installation; refusing to upgrade.')
    sessions = json.loads(result.stdout).get('sessions', [])
    if any(s['state'] in ('starting', 'recording', 'queued', 'transcribing', 'auditing') for s in sessions):
        raise SystemExit('Capture or processing is active. Finish it before installing.')
if (root / 'active.json').exists():
    raise SystemExit('An active session pointer exists. Finish or recover it before installing.')
PY
if [[ -n "${CALLREC_UV:-}" ]]; then
  UV_BIN="$CALLREC_UV"
elif command -v uv >/dev/null; then
  UV_BIN="$(command -v uv)"
else
  "$PYTHON_BIN" -m venv "$APP_DIR/bootstrap"
  "$APP_DIR/bootstrap/bin/python" -m pip install 'uv==0.13.5'
  UV_BIN="$APP_DIR/bootstrap/bin/uv"
fi
RELEASE_DIR="$(mktemp -d "$APP_DIR/app/releases/callrec.XXXXXXXX")"
cp "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$RELEASE_DIR/"
cp -R "$PROJECT_DIR/src" "$RELEASE_DIR/src"
"$UV_BIN" sync --project "$RELEASE_DIR" --locked --no-editable --python "$PYTHON_BIN"
"$PYTHON_BIN" - "$RELEASE_DIR" "$HOME/.local/bin/callrec" <<'PY'
import os, pathlib, shlex, sys, tempfile
project, target = map(pathlib.Path, sys.argv[1:])
launcher = '#!/bin/sh\n# callrec-managed-launcher\nexec ' + shlex.quote(str(project / '.venv/bin/callrec')) + ' "$@"\n'
if target.exists() or target.is_symlink():
    if target.is_symlink() or 'callrec-managed-launcher' not in target.read_text():
        raise SystemExit(f'Refusing to overwrite unrelated launcher: {target}')
fd, name = tempfile.mkstemp(dir=target.parent, prefix='.callrec-')
with os.fdopen(fd, 'w') as f:
    f.write(launcher)
    f.flush()
    os.fsync(f.fileno())
os.chmod(name, 0o755)
os.replace(name, target)
print(f'Installed {target}. Next: callrec setup')
PY
