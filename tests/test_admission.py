import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

try:
    import cv2
    import httpx
    import anyio
    import numpy as np
    from trashsmart.pipeline import main as api
    from trashsmart.pipeline.admission import PredictionAdmission
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error))


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_has_separate_capacity_when_default_pool_is_full(self):
        started, release = threading.Event(), threading.Event()

        def blocked(*args, **kwargs):
            started.set()
            release.wait(5)
            return {"count": 0}

        limiter = anyio.to_thread.current_default_thread_limiter()
        original = limiter.total_tokens
        limiter.total_tokens = 1
        data = cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes()
        try:
            with patch.object(api, "get_models", return_value=(Mock(), Mock())), \
                    patch.object(api, "process_image", side_effect=blocked):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                    task = asyncio.create_task(client.post("/predict-image", files={"file": ("a.jpg", data)}))
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 3))
                        for route in ("/health", "/live"):
                            response = await asyncio.wait_for(client.get(route), 2)
                            self.assertEqual(response.status_code, 200)
                    finally:
                        release.set()
                        await task
        finally:
            limiter.total_tokens = original

    async def test_overload_rejected_while_health_and_liveness_respond(self):
        started, release, lock = threading.Event(), threading.Event(), threading.Lock()
        count = 0

        def blocked(*args, **kwargs):
            nonlocal count
            with lock:
                count += 1
                if count == 2:
                    started.set()
            release.wait(5)
            return {"count": 0}

        data = cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes()
        guarded = PredictionAdmission(api.app, limit=2)
        with patch.object(api, "get_models", return_value=(Mock(), Mock())), \
                patch.object(api, "process_image", side_effect=blocked):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://test") as client:
                tasks = [asyncio.create_task(client.post("/predict-image", files={"file": ("a.jpg", data)}))
                         for _ in range(2)]
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
                    response = await asyncio.wait_for(client.post("/predict-image", files={"file": ("a.jpg", data)}), 2)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.headers["Retry-After"], "1")
                    for route in ("/live", "/health"):
                        self.assertEqual((await asyncio.wait_for(client.get(route), 2)).status_code, 200)
                finally:
                    release.set()
                    results = await asyncio.gather(*tasks)
                self.assertTrue(all(result.status_code == 200 for result in results))

    async def test_large_body_rejected_before_multipart_parse(self):
        guarded = PredictionAdmission(api.app, max_body=10)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://test") as client:
            response = await client.post("/predict-image", content=b"x" * 20)
            self.assertEqual(response.status_code, 413)
            self.assertEqual((await client.get("/live")).status_code, 200)

    async def test_chunked_body_limit_without_content_length(self):
        async def chunks():
            yield b"x" * 8
            yield b"y" * 8

        guarded = PredictionAdmission(api.app, max_body=10)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded), base_url="http://test") as client:
            response = await client.post("/predict-image-url", content=chunks(),
                                         headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)
