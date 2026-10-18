"""Test checkpoint publishing without installing ML packages or training."""
import contextlib
import importlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class CheckpointWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.active = self.root / "models" / "best_new.pt"
        self.active.parent.mkdir()
        self.active.write_bytes(b"original weights")
        self.run = self.root / "runs" / "finetune"
        self.metrics = types.SimpleNamespace(box=types.SimpleNamespace(
            mp=0.9, mr=0.8, map=0.6, map50=0.85, map75=0.7))
        modules = {
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            "dotenv": types.SimpleNamespace(load_dotenv=lambda *a: None),
            "ultralytics": types.SimpleNamespace(YOLO=None),
        }
        with patch.dict("sys.modules", modules):
            self.module = importlib.import_module("trashsmart.yolo_training.train_yolo")

    def run_training(self, failure=None):
        calls = []
        test = self

        class FakeYOLO:
            def __init__(self, path):
                calls.append(str(path))
                self.trainer = types.SimpleNamespace(save_dir=test.run)

            def train(self, **kwargs):
                if failure == "training":
                    raise RuntimeError("training failed")
                checkpoint = test.run / "weights" / "best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"trained weights")
                return "trained"

            def val(self, **kwargs):
                if failure == "validation":
                    raise RuntimeError("validation failed")
                return test.metrics

        with patch.multiple(self.module, YOLO=FakeYOLO,
                            YOLO_MODEL_PATH=self.active,
                            PROJECT_NAME=str(self.root / "runs")), \
                patch.object(self.module, "download_and_extract_dataset"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.module.main()
        return calls

    def test_success_publishes_validated_checkpoint_and_preserves_backup(self):
        calls = self.run_training()
        self.assertEqual(calls, [str(self.active), str(self.run / "weights" / "best.pt")])
        self.assertEqual(self.active.read_bytes(), b"trained weights")
        backups = list((self.root / "runs" / "backups").glob("*.pt"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"original weights")
        self.assertEqual(list(self.active.parent.glob("best_new_*.pt")), [])

    def test_training_failure_keeps_active_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.run_training("training")
        self.assertEqual(self.active.read_bytes(), b"original weights")

    def test_validation_failure_keeps_active_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "validation failed"):
            self.run_training("validation")
        self.assertEqual(self.active.read_bytes(), b"original weights")


if __name__ == "__main__":
    unittest.main()
