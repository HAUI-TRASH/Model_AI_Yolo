"""Bounded image decoding and URL fetching with DNS-pinned public connections."""
import io
import ipaddress
import os
import socket
import time
from urllib.parse import urljoin, urlsplit

import certifi
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import urllib3


def positive_setting(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


MAX_IMAGE_BYTES = positive_setting("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
MAX_IMAGE_PIXELS = positive_setting("MAX_IMAGE_PIXELS", 12_000_000)
MAX_IMAGE_SIDE = positive_setting("MAX_IMAGE_SIDE", 8192)
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 1024 * 1024
MAX_URL_REDIRECTS = 3
URL_TIMEOUT = 15


class ImageInputError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def decode_image(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError("Image exceeds byte limit", 413)
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG", "PNG", "WEBP", "BMP")) as header:
            width, height = header.size
            if (width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS):
                raise ImageInputError("Image exceeds pixel/dimension limit", 413)
            if header.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ImageInputError("Supported images: JPEG, PNG, WEBP, BMP")
            header.verify()
    except Image.DecompressionBombError as error:
        raise ImageInputError("Image exceeds decoder safety limit", 413) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        if isinstance(error, ImageInputError):
            raise
        raise ImageInputError("Invalid image") from error
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageInputError("Invalid image")
    return image


def read_upload(file):
    # Read at most one byte beyond the limit, even if Content-Length was absent.
    return decode_image(file.read(MAX_IMAGE_BYTES + 1))


def public_target(url):
    parsed = urlsplit(url)
    try:
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except (ValueError, UnicodeError) as error:
        raise ImageInputError("Invalid image URL") from error
    if (parsed.scheme not in {"http", "https"} or not host or parsed.username is not None
            or parsed.password is not None or port != (443 if parsed.scheme == "https" else 80)):
        raise ImageInputError("Only HTTP/HTTPS URLs with standard ports and no credentials are allowed")
    allowed = {item.strip().lower() for item in os.getenv("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if item.strip()}
    if allowed and host not in allowed:
        raise ImageInputError("Image URL host is not allowed")
    try:
        addresses = {row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except socket.gaierror as error:
        raise ImageInputError("Cannot resolve image URL host") from error
    if not addresses:
        raise ImageInputError("Cannot resolve image URL host")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        ip = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise ImageInputError("Image URL must resolve only to public IP addresses")
    return parsed, host, port, sorted(addresses)[0]


def download_image(url):
    deadline = time.monotonic() + URL_TIMEOUT
    for redirect in range(MAX_URL_REDIRECTS + 1):
        parsed, host, port, address = public_target(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ImageInputError("Image download timed out")
        kwargs = dict(host=address, port=port, maxsize=1, block=True,
                      timeout=urllib3.Timeout(connect=min(5, remaining), read=min(5, remaining)))
        # Connect to the validated numeric IP: never resolve the host a second
        # time. HTTPS still verifies the original host and sends correct SNI.
        pool = (urllib3.HTTPSConnectionPool(**kwargs, server_hostname=host, assert_hostname=host,
                                           cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
                if parsed.scheme == "https" else urllib3.HTTPConnectionPool(**kwargs))
        response = None
        try:
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            host_header = f"[{host}]" if ":" in host else host
            response = pool.urlopen("GET", target, headers={"Host": host_header, "Accept-Encoding": "identity"},
                                    redirect=False, retries=False, preload_content=False)
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location or redirect == MAX_URL_REDIRECTS:
                    raise ImageInputError("Too many or invalid image URL redirects")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ImageInputError("Image URL did not return HTTP 200")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ImageInputError("Compressed HTTP responses are not supported")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    if int(length) < 0 or int(length) > MAX_IMAGE_BYTES:
                        raise ImageInputError("Image exceeds byte limit", 413)
                except ValueError as error:
                    if isinstance(error, ImageInputError):
                        raise
                    raise ImageInputError("Invalid Content-Length") from error
            content = bytearray()
            while True:
                if time.monotonic() >= deadline:
                    raise ImageInputError("Image download timed out")
                chunk = response.read1(min(64 * 1024, MAX_IMAGE_BYTES + 1 - len(content)), decode_content=False)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise ImageInputError("Image exceeds byte limit", 413)
            return decode_image(content)
        except urllib3.exceptions.HTTPError as error:
            raise ImageInputError("Cannot download image") from error
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ImageInputError("Too many redirects")
