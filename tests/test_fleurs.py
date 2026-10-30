import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.fleurs import extract_selected, rows_for_language


class FleursTests(unittest.TestCase):
    def test_selects_disjoint_cases_from_verified_metadata(self):
        rows = "".join(f"{i}\t{i}.wav\tनमस्ते {i}\n" for i in reversed(range(60)))
        data = rows.encode()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        with patch("benchmarks.fleurs.urllib.request.urlopen", return_value=io.BytesIO(data)):
            result = rows_for_language("hi_in", blob)
        self.assertEqual(len(result), 50)
        self.assertEqual(result[0][1], "0.wav")
        self.assertEqual(result[49][1], "49.wav")
        with patch("benchmarks.fleurs.urllib.request.urlopen", return_value=io.BytesIO(data)):
            with self.assertRaisesRegex(ValueError, "metadata"):
                rows_for_language("hi_in", "wrong")

    def test_extract_never_writes_archive_path_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "test.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                payload = b"audio fixture"
                item = tarfile.TarInfo("../../example.wav")
                item.size = len(payload)
                tar.addfile(item, io.BytesIO(payload))
            output = root / "safe"
            extract_selected(archive, output, [["1", "example.wav", "text"]])
            self.assertEqual((output / "example.wav").read_bytes(), payload)
            (output / "example.wav").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "differs"):
                extract_selected(archive, output, [["1", "example.wav", "text"]])
            self.assertEqual((output / "example.wav").read_bytes(), b"different")


if __name__ == "__main__":
    unittest.main()
