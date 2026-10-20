"""Compare actual evaluation and deployed inference tensors, without model weights."""
import pickle
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from trashsmart.shared import preprocessing
    from trashsmart.shared.transforms import get_eval_transform
    from trashsmart.training.data_utils import get_transforms
    from trashsmart.pipeline.utils import preprocess_for_resnet, preprocess_batch_for_resnet
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"Preprocessing tests require ML dependencies: {error}")


class ResNetPreprocessingTests(unittest.TestCase):
    def test_validation_and_api_produce_identical_tensors(self):
        rng = np.random.default_rng(42)
        _, validation_transform = get_transforms()
        crops = [
            rng.integers(0, 256, (70, 200, 3), dtype=np.uint8),
            rng.integers(0, 256, (240, 80, 3), dtype=np.uint8),
            rng.integers(0, 40, (60, 100, 3), dtype=np.uint8),
            cv2.GaussianBlur(rng.integers(0, 256, (90, 110, 3), dtype=np.uint8), (15, 15), 4),
            np.full((1, 1000, 3), 100, dtype=np.uint8),
        ]
        batch = preprocess_batch_for_resnet(crops).cpu()
        for index, crop in enumerate(crops):
            with self.subTest(shape=crop.shape):
                original = crop.copy()
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                expected = validation_transform(pil)
                self.assertTrue(torch.equal(expected, preprocess_for_resnet(crop).cpu()[0]))
                self.assertTrue(torch.equal(expected, batch[index]))
                np.testing.assert_array_equal(crop, original)
                self.assertTrue(torch.isfinite(expected).all())
                self.assertEqual(expected.shape, (3, preprocessing.RESNET_INPUT_SIZE,
                                                  preprocessing.RESNET_INPUT_SIZE))

    def test_enhancement_is_called_once_per_crop(self):
        crop = np.full((60, 100, 3), 10, dtype=np.uint8)
        with patch.object(preprocessing, "quick_enhance_for_resnet",
                          wraps=preprocessing.quick_enhance_for_resnet) as enhance:
            preprocess_batch_for_resnet([crop, crop])
        self.assertEqual(enhance.call_count, 2)

    def test_transform_is_picklable_for_windows_workers(self):
        transform = get_eval_transform()
        restored = pickle.loads(pickle.dumps(transform))
        image = Image.new("RGB", (40, 100), (20, 70, 200))
        self.assertTrue(torch.equal(transform(image), restored(image)))
        self.assertEqual(restored(image).device.type, "cpu")

    def test_letterbox_preserves_entire_image_and_adds_white_padding(self):
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        crop[:, :10] = (0, 0, 255)
        output = preprocessing.letterbox_to_square(crop, 100)
        np.testing.assert_array_equal(output[25:75], crop)
        self.assertTrue((output[:25] == 255).all())
        self.assertTrue((output[75:] == 255).all())

    def test_empty_crop_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocessing.preprocess_resnet_crop(np.zeros((0, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
