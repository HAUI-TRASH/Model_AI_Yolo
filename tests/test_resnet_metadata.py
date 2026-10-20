import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    import torch
    from trashsmart.shared import model as model_utils
    from trashsmart.training import train
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error))

from trashsmart.shared import classes


class ResNetMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "model.pth"
        self.model = torch.nn.Linear(2, len(classes.CLASS_NAMES))

    def metadata(self):
        return {"format_version": 1, "architecture": "TrashResNet50",
                "class_names": list(classes.CLASS_NAMES), "num_classes": len(classes.CLASS_NAMES),
                "model_state_dict": self.model.state_dict()}

    def load(self):
        with patch.object(model_utils, "build_trash_resnet50",
                          side_effect=lambda **kwargs: torch.nn.Linear(2, len(classes.CLASS_NAMES))):
            return model_utils.load_resnet_checkpoint(self.path, len(classes.CLASS_NAMES), torch.device("cpu"))

    def test_training_saves_metadata_and_loader_round_trips_weights(self):
        with patch.object(train, "BEST_MODEL_PATH", self.path):
            self.assertEqual(train._save_if_best(self.model, 0.96, 0.95), 0.96)
        checkpoint = torch.load(self.path, weights_only=True)
        self.assertEqual(checkpoint["class_names"], classes.CLASS_NAMES)
        self.assertEqual(checkpoint["val_accuracy"], 0.96)
        restored = self.load()
        self.assertFalse(restored.training)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]))

    def test_same_class_count_but_wrong_order_rejected_before_model_build(self):
        checkpoint = self.metadata()
        checkpoint["class_names"] = list(reversed(classes.CLASS_NAMES))
        torch.save(checkpoint, self.path)
        with patch.object(model_utils, "build_trash_resnet50") as builder:
            with self.assertRaisesRegex(ValueError, "class order"):
                model_utils.load_resnet_checkpoint(self.path, len(classes.CLASS_NAMES), torch.device("cpu"))
        builder.assert_not_called()

    def test_legacy_weights_load_with_original_class_order(self):
        torch.save(self.model.state_dict(), self.path)
        restored = self.load()
        self.assertTrue(torch.equal(self.model.weight, restored.weight))

    def test_legacy_weights_reject_changed_configured_order(self):
        torch.save(self.model.state_dict(), self.path)
        with patch.object(classes, "CLASS_NAMES", list(reversed(classes.CLASS_NAMES))):
            with self.assertRaisesRegex(ValueError, "Legacy"):
                self.load()

    def test_missing_or_inconsistent_metadata_is_rejected(self):
        for key, value in (("num_classes", 5), ("class_names", None),
                           ("format_version", 2), ("architecture", "OtherModel")):
            with self.subTest(key=key):
                checkpoint = self.metadata()
                checkpoint[key] = value
                with self.assertRaises(ValueError):
                    classes.validate_checkpoint_classes(checkpoint, len(classes.CLASS_NAMES))
        checkpoint = self.metadata()
        del checkpoint["class_names"]
        with self.assertRaises(ValueError):
            classes.validate_checkpoint_classes(checkpoint, len(classes.CLASS_NAMES))

    def test_invalid_accuracy_never_replaces_existing_checkpoint(self):
        self.path.write_bytes(b"original checkpoint")
        for value in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1):
            with self.subTest(value=value), patch.object(train, "BEST_MODEL_PATH", self.path):
                with self.assertRaises((FloatingPointError, ValueError)):
                    train._save_if_best(self.model, value, 0.95)
                self.assertEqual(self.path.read_bytes(), b"original checkpoint")

    def test_invalid_best_accuracy_rejected(self):
        with patch.object(train, "save_checkpoint_atomic") as save:
            with self.assertRaises(FloatingPointError):
                train._save_if_best(self.model, 0.96, float("nan"))
        save.assert_not_called()
