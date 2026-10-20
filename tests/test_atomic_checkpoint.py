from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from trashsmart.shared.checkpoints import save_checkpoint_atomic


class AtomicCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "best_resnet50_model1.pth"
        self.path.write_bytes(b"original checkpoint")

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_active_checkpoint_stays_old_until_serialization_finishes(self):
        state = {"weights": [1, 2, 3]}

        def save(data, stream):
            self.assertEqual(self.path.read_bytes(), b"original checkpoint")
            pickle.dump(data, stream)
            stream.flush()
            self.assertEqual(self.path.read_bytes(), b"original checkpoint")

        save_checkpoint_atomic(state, self.path, save)
        self.assertEqual(pickle.loads(self.path.read_bytes()), state)
        self.assert_no_temporary_files()

    def test_partial_write_failure_preserves_old_checkpoint(self):
        def fail(data, stream):
            stream.write(b"partial new checkpoint")
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            save_checkpoint_atomic({}, self.path, fail)
        self.assertEqual(self.path.read_bytes(), b"original checkpoint")
        self.assert_no_temporary_files()

    def test_replace_failure_preserves_old_checkpoint(self):
        with patch("trashsmart.shared.checkpoints.os.replace", side_effect=PermissionError("busy")):
            with self.assertRaises(PermissionError):
                save_checkpoint_atomic({}, self.path, pickle.dump)
        self.assertEqual(self.path.read_bytes(), b"original checkpoint")
        self.assert_no_temporary_files()

    def test_flush_failure_preserves_old_checkpoint(self):
        with patch("trashsmart.shared.checkpoints.os.fsync", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                save_checkpoint_atomic({}, self.path, pickle.dump)
        self.assertEqual(self.path.read_bytes(), b"original checkpoint")
        self.assert_no_temporary_files()

    def test_first_save_creates_parent_directory(self):
        destination = self.root / "new_models" / "best.pth"
        save_checkpoint_atomic({"weights": 1}, destination, pickle.dump)
        self.assertEqual(pickle.loads(destination.read_bytes()), {"weights": 1})
        self.assertEqual(list(destination.parent.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
