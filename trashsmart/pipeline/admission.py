"""Reject overload and oversized bodies before multipart parsing or thread dispatch."""
import threading
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from trashsmart.pipeline.input_limits import MAX_REQUEST_BYTES, positive_setting

PREDICTION_PATHS = {"/predict-image", "/predict-image-json", "/predict-image-url"}


class BodyTooLarge(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="Request body exceeds byte limit")


class PredictionAdmission:
    def __init__(self, app, limit=None, max_body=MAX_REQUEST_BYTES):
        self.app = app
        self.slots = threading.BoundedSemaphore(
            limit if limit is not None else positive_setting("MAX_CONCURRENT_PREDICTIONS", 2))
        self.max_body = max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "").rstrip("/") not in PREDICTION_PATHS:
            return await self.app(scope, receive, send)
        if not self.slots.acquire(blocking=False):
            return await JSONResponse({"error": "Prediction service busy; retry later"}, status_code=503,
                                      headers={"Retry-After": "1"})(scope, receive, send)
        try:
            length = dict(scope.get("headers", [])).get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    return await JSONResponse({"error": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
                if declared < 0 or declared > self.max_body:
                    raise BodyTooLarge
            total = 0

            async def limited_receive():
                nonlocal total
                message = await receive()
                if message["type"] == "http.request":
                    total += len(message.get("body", b""))
                    if total > self.max_body:
                        raise BodyTooLarge
                return message

            await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            await JSONResponse({"error": "Request body exceeds byte limit"}, status_code=413)(scope, receive, send)
        finally:
            self.slots.release()
