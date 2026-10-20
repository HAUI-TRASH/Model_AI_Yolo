import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

from trashsmart.shared.classes import CLASS_NAMES, validate_class_order
from trashsmart.shared.metrics import require_finite


class EarlyStoppingTests(unittest.TestCase):
    def make_stop(self, **kwargs):
        source = Path(__file__).resolve().parents[1] / "trashsmart/training/train.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EarlyStopping")
        env = {"copy": copy, "require_finite": require_finite}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
        return env["EarlyStopping"](**kwargs)

    def test_flat_loss_stops_at_patience(self):
        stop, model = self.make_stop(patience=2), Mock()
        stop(1.0, model)
        stop(1.0, model)
        self.assertFalse(stop.early_stop)
        stop(1.0, model)
        self.assertTrue(stop.early_stop)

    def test_improvement_resets_counter(self):
        stop, model = self.make_stop(patience=2), Mock()
        for loss in (1.0, 1.1, 0.9, 0.95):
            stop(loss, model)
        self.assertFalse(stop.early_stop)
        self.assertEqual(stop.counter, 1)
        stop(0.95, model)
        self.assertTrue(stop.early_stop)

    def test_min_delta_requires_strict_improvement(self):
        stop, model = self.make_stop(patience=2, min_delta=0.1), Mock()
        for loss in (1.0, 0.99, 0.98):
            stop(loss, model)
        self.assertTrue(stop.early_stop)

    def test_max_mode_counts_plateau(self):
        stop, model = self.make_stop(patience=1, mode="max"), Mock()
        stop(0.8, model)
        stop(0.9, model)
        self.assertFalse(stop.early_stop)
        stop(0.9, model)
        self.assertTrue(stop.early_stop)

    def test_invalid_metric_leaves_early_stopping_state_unchanged(self):
        stop, model = self.make_stop(patience=2), Mock()
        stop(1.0, model)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                stop(value, model)
            self.assertEqual(stop.best_score, -1.0)
            self.assertEqual(stop.counter, 0)
            self.assertFalse(stop.early_stop)


class ClassOrderTests(unittest.TestCase):
    def test_correct_classes_accepted(self):
        validate_class_order(CLASS_NAMES)

    def test_reordered_classes_rejected(self):
        with self.assertRaises(ValueError):
            validate_class_order(list(reversed(CLASS_NAMES)))

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            validate_class_order(CLASS_NAMES[:-1])

    def test_training_rejects_wrong_order_before_building_or_saving_model(self):
        source = Path(__file__).resolve().parents[1] / "trashsmart/training/main.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        build, train = Mock(), Mock()
        env = {
            "DIR_PATH": SimpleNamespace(exists=lambda: True), "NUM_CLASSES": len(CLASS_NAMES),
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "get_data_loaders": lambda: (None, None, list(reversed(CLASS_NAMES))),
            "validate_class_order": validate_class_order, "build_model": build,
            "train_two_stages": train,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
        with self.assertRaises(ValueError):
            env["main"]()
        build.assert_not_called()
        train.assert_not_called()
