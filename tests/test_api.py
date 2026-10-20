import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import cv2
    import httpx
    import numpy as np
    import torch
    from fastapi.testclient import TestClient
    from trashsmart.pipeline import main as api
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error))


def image_bytes():
    return cv2.imencode(".jpg", np.zeros((40, 40, 3), dtype=np.uint8))[1].tobytes()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_readiness_failure_and_liveness(self):
        with patch.object(api, "get_models", side_effect=RuntimeError("bad checkpoint")):
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.assertEqual(self.client.get("/live").status_code, 200)

    def test_readiness_success(self):
        with patch.object(api, "get_models", return_value=(Mock(), Mock())):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["modelsReady"])

    def test_invalid_upload_returns_400_for_both_routes(self):
        for route in ("/predict-image", "/predict-image-json"):
            with self.subTest(route=route):
                response = self.client.post(route, files={"file": ("bad.jpg", b"invalid")})
                self.assertEqual(response.status_code, 400)

    def test_model_load_error_returns_json_for_both_routes(self):
        with patch.object(api, "get_models", side_effect=RuntimeError("bad checkpoint")):
            for route in ("/predict-image", "/predict-image-json"):
                response = self.client.post(route, files={"file": ("image.jpg", image_bytes())})
                self.assertEqual(response.status_code, 500)
                self.assertIn("error", response.json())

    def test_upload_response_and_request_id(self):
        result = {"count": 0, "detections": []}
        with patch.object(api, "get_models", return_value=(Mock(), Mock())), \
                patch.object(api, "process_image", return_value=result) as process:
            response = self.client.post("/predict-image-json?request_id=12",
                                        files={"file": ("image.jpg", image_bytes())})
        self.assertEqual(response.json(), result)
        self.assertEqual(process.call_args.kwargs["request_id"], 12)

    def test_url_parameters_are_forwarded(self):
        with patch.object(api, "download_image_to_cv2", return_value=np.zeros((40, 40, 3), np.uint8)), \
                patch.object(api, "get_models", return_value=(Mock(), Mock())), \
                patch.object(api, "process_image", return_value={"count": 0}) as process:
            response = self.client.post("/predict-image-url", json={
                "image_url": "https://example.com/image.jpg", "conf": 0.7, "iou": 0.4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(process.call_args.kwargs["conf"], 0.7)
        self.assertEqual(process.call_args.kwargs["iou"], 0.4)

    def test_invalid_conf_rejected(self):
        response = self.client.post("/predict-image-url", json={
            "image_url": "https://example.com/image.jpg", "conf": 2})
        self.assertEqual(response.status_code, 422)

    def test_concurrent_initialization_loads_once(self):
        with patch.object(api, "_yolo_model", None), patch.object(api, "_resnet_model", None), \
                patch.object(api, "load_yolo_model", return_value=Mock()) as yolo, \
                patch.object(api, "load_resnet_model", return_value=Mock()) as resnet:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: api.get_models(), range(8)))
        yolo.assert_called_once()
        resnet.assert_called_once()

    def test_confirmation_uses_configured_threshold(self):
        box = SimpleNamespace(conf=torch.tensor(0.9), cls=torch.tensor(0),
                              xyxy=torch.tensor([[0, 0, 40, 40]]))
        yolo = Mock(return_value=[SimpleNamespace(boxes=[box], names={0: "Bottle"})])
        with patch.object(api, "RESNET_CONF_THRESHOLD", 0.7), \
                patch.object(api, "get_class_labels_batch", return_value=[("unknown", 0.6)]), \
                patch.object(api, "upload_image_bytes_to_cloudinary", return_value="https://example.com/a.jpg"), \
                patch.object(api, "upload_crop_to_cloudinary", return_value="https://example.com/c.jpg"):
            result = api.process_image(np.zeros((40, 40, 3), np.uint8), yolo,
                                       lambda batch: torch.zeros((1, 4)))
        self.assertTrue(result["requiresConfirmation"])
        self.assertTrue(result["detections"][0]["needsConfirm"])


class ApiConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_does_not_block_liveness_on_same_event_loop(self):
        started, release = threading.Event(), threading.Event()

        def blocked_process(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test timeout")
            return {"count": 0}

        transport = httpx.ASGITransport(app=api.app)
        with patch.object(api, "get_models", return_value=(Mock(), Mock())), \
                patch.object(api, "process_image", side_effect=blocked_process):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                upload = asyncio.create_task(client.post("/predict-image", files={
                    "file": ("image.jpg", image_bytes())}))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
                    response = await asyncio.wait_for(client.get("/live"), timeout=2)
                    self.assertEqual(response.status_code, 200)
                finally:
                    release.set()
                    result = await upload
                self.assertEqual(result.status_code, 200)
