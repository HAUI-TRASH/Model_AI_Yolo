"""Exercise real stage orchestration with simulated epochs; no ML dependencies."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from trashsmart.shared.classes import CLASS_NAMES
from trashsmart.shared.metrics import require_finite, require_accuracy


class ResNetCheckpointTests(unittest.TestCase):
    def run_stages(self, accuracies, validation_losses=None):
        # Load the actual training functions without importing optional ML packages.
        source = Path(__file__).resolve().parents[1] / "trashsmart/training/train.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        saved = []
        self.saved = saved
        model = Mock()
        model.state_dict.side_effect = lambda: model.epoch
        losses = validation_losses if validation_losses is not None else [1.0] * len(accuracies)
        validation = iter(enumerate(zip(accuracies, losses), start=1))

        def validate(*args):
            epoch, (accuracy, loss) = next(validation)
            model.epoch = epoch
            return loss, accuracy, [], []

        env = {
            "save_checkpoint_atomic": lambda state, path, save_fn: save_fn(state, path),
            "torch": SimpleNamespace(save=lambda state, path: saved.append(state["model_state_dict"])),
            "CLASS_NAMES": CLASS_NAMES, "require_finite": require_finite,
            "require_accuracy": require_accuracy,
            "MODELS_DIR": Mock(), "BEST_MODEL_PATH": "unused.pth",
            "nn": SimpleNamespace(CrossEntropyLoss=Mock()),
            "optim": SimpleNamespace(Adam=Mock()),
            "DEVICE": "cpu", "LEARNING_RATE": 0.0001,
            "STAGE1_EPOCHS": 2, "EPOCHS": 2,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), env)
        env.update({
            "train_epoch": lambda *args: (1.0, 0.5, [], []),
            "validate_epoch": validate,
            "_make_scheduler": lambda *args: Mock(),
            "EarlyStopping": lambda **kwargs: Mock(early_stop=False),
        })
        with contextlib.redirect_stdout(io.StringIO()):
            history = env["train_two_stages"](model, [], [], [])
        return saved, history

    def test_worse_stage2_preserves_best_stage1(self):
        saved, history = self.run_stages([0.95, 0.90, 0.60, 0.80])
        self.assertEqual(saved, [1])
        self.assertEqual(history["val_accuracy"], [0.95, 0.90, 0.60, 0.80])

    def test_better_stage2_replaces_checkpoint_only_on_improvement(self):
        saved, _ = self.run_stages([0.95, 0.90, 0.96, 0.94])
        self.assertEqual(saved, [1, 3])

    def test_equal_accuracy_does_not_replace_checkpoint(self):
        saved, _ = self.run_stages([0.95, 0.90, 0.95, 0.95])
        self.assertEqual(saved, [1])

    def test_first_checkpoint_is_saved_even_at_zero_accuracy(self):
        saved, _ = self.run_stages([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(saved, [1])

    def test_nan_accuracy_in_stage2_preserves_previous_checkpoint(self):
        with self.assertRaises(FloatingPointError):
            self.run_stages([0.95, 0.90, float("nan"), 0.96])
        self.assertEqual(self.saved, [1])

    def test_nonfinite_validation_loss_does_not_publish_even_with_better_accuracy(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                self.run_stages([0.95, 0.90, 0.99, 0.99], [1.0, 1.0, value, 1.0])
            self.assertEqual(self.saved, [1])


if __name__ == "__main__":
    unittest.main()
