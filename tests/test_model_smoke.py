"""Opt-in real-model smoke tests. Generated speech is not real lesson validation."""

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from callrec.cli import import_audio, wait
from callrec.common import model_details, read_json, root_path


@unittest.skipUnless(os.environ.get("CALLREC_MODEL_SMOKE") == "1", "opt-in local model tests")
class ModelSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.root = cls.base / "data"
        (cls.root / "models").mkdir(parents=True)
        for engine in ("whisper", "qwen"):
            name = model_details(engine)["directory"]
            (cls.root / "models" / name).symlink_to(root_path() / "models" / name)
        samples = {
            "hindi": (
                "Lekha",
                "नमस्ते। आज हम हिन्दी सीख रहे हैं। मुझे हिन्दी बोलना पसंद है। कल मैं अपने दोस्त के साथ बाज़ार गया था। यह रिकॉर्डिंग का आखिरी वाक्य है।",
            ),
            "french": (
                "Thomas",
                "Bonjour. Aujourd’hui nous parlons français. Je suis allé au marché avec mon ami. Ceci est la dernière phrase de cet enregistrement.",
            ),
            "english": (
                "Daniel",
                "Now we switch to English. Please preserve the original language. This is the final English sentence.",
            ),
        }
        for language, (voice, text) in samples.items():
            subprocess.run(
                ["say", "-v", voice, "-o", str(cls.base / f"{language}.aiff"), text], check=True
            )
        for language in ("hindi", "french"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(cls.base / f"{language}.aiff"),
                    "-i",
                    str(cls.base / "english.aiff"),
                    "-filter_complex",
                    "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                    "-map",
                    "[out]",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(cls.base / f"{language}.wav"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-stream_loop",
                    "2",
                    "-i",
                    str(cls.base / f"{language}.wav"),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(cls.base / f"{language}-long.wav"),
                ],
                check=True,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def transcribe(self, file, profile, engine):
        with (
            patch(
                "socket.socket.connect", side_effect=RuntimeError("Offline test blocked a socket")
            ) as connect,
            patch(
                "socket.socket.connect_ex",
                side_effect=RuntimeError("Offline test blocked a socket"),
            ) as connect_ex,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            before = set((self.root / "sessions").glob("*"))
            import_audio(self.root, self.base / file, profile, engine)
            session = (set((self.root / "sessions").glob("*")) - before).pop()
            self.assertEqual(wait(self.root, session.name), 0)
            report = read_json(session / "transcript.json")
            connect.assert_not_called()
            connect_ex.assert_not_called()
        self.assertTrue(report["processing_complete"])
        self.assertEqual(report["expected_windows"], report["processed_windows"])
        metadata = read_json(session / "session.json")
        attempt = read_json(session / "attempts" / f"{metadata['attempt_id']}.json")
        self.assertTrue(attempt["model_process"]["network_guard"])
        self.assertEqual(attempt["model_process"]["blocked_network_attempts"], 0)
        return " ".join(s["text"] for s in report["segments"])

    def test_hindi_english_short_and_across_windows(self):
        for suffix, count in (("", 1), ("-long", 3)):
            text = self.transcribe(f"hindi{suffix}.wav", "hindi", "qwen")
            self.assertEqual(text.lower().count("this is the final english sentence"), count)
            self.assertEqual(text.lower().count("now we switch to english"), count)
            self.assertEqual(text.count("बाजार"), count)
            self.assertGreater(sum("\u0900" <= c <= "\u097f" for c in text), 80 * count)

    def test_french_english_short_and_across_windows(self):
        for suffix, count in (("", 1), ("-long", 3)):
            text = self.transcribe(f"french{suffix}.wav", "french", "whisper")
            self.assertEqual(text.lower().count("this is the final english sentence"), count)
            self.assertEqual(text.count("dernière phrase"), count)
            self.assertEqual(text.count("marché"), count)

    def test_general_english(self):
        text = self.transcribe("english.aiff", "general", "whisper")
        self.assertIn("This is the final English sentence", text)


if __name__ == "__main__":
    unittest.main()
