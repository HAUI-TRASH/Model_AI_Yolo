import io
import socket
import unittest
from unittest.mock import Mock, patch

try:
    import cv2
    import numpy as np
    from trashsmart.pipeline import input_limits as limits
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error))


def address(ip):
    return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class Response:
    def __init__(self, data=b"", status=200, headers=None):
        self.body = io.BytesIO(data)
        self.status = status
        self.headers = headers or {}

    def read1(self, amount, **kwargs):
        return self.body.read(amount)

    def close(self):
        self.body.close()


class ImageLimitsTests(unittest.TestCase):
    def jpeg(self):
        return cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes()

    def test_valid_image(self):
        self.assertEqual(limits.decode_image(self.jpeg()).shape, (40, 40, 3))

    def test_empty_or_invalid_data(self):
        for data in (b"", b"not an image"):
            with self.assertRaises(limits.ImageInputError):
                limits.decode_image(data)

    def test_upload_reads_only_one_byte_beyond_limit(self):
        source = io.BytesIO(b"x" * 1000)
        with patch.object(limits, "MAX_IMAGE_BYTES", 100):
            with self.assertRaises(limits.ImageInputError) as error:
                limits.read_upload(source)
        self.assertEqual(error.exception.status_code, 413)
        self.assertEqual(source.tell(), 101)

    def test_pixel_limit_is_checked_before_opencv_decode(self):
        with patch.object(limits, "MAX_IMAGE_PIXELS", 100), patch.object(limits.cv2, "imdecode") as decode:
            with self.assertRaises(limits.ImageInputError) as error:
                limits.decode_image(self.jpeg())
        self.assertEqual(error.exception.status_code, 413)
        decode.assert_not_called()


class UrlSafetyTests(unittest.TestCase):
    def test_private_loopback_linklocal_multicast_and_mapped_addresses_blocked(self):
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1",
                   "::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"):
            with self.subTest(ip=ip), patch.object(limits.socket, "getaddrinfo", return_value=address(ip)):
                with self.assertRaises(limits.ImageInputError):
                    limits.public_target("https://example.com/image.jpg")

    def test_mixed_public_and_private_dns_answers_blocked(self):
        with patch.object(limits.socket, "getaddrinfo", return_value=address("8.8.8.8") + address("127.0.0.1")):
            with self.assertRaises(limits.ImageInputError):
                limits.public_target("https://example.com/image.jpg")

    def test_credentials_protocol_and_nonstandard_port_blocked(self):
        for url in ("file:///etc/passwd", "http://user:password@example.com/a", "https://example.com:8443/a"):
            with self.subTest(url=url), self.assertRaises(limits.ImageInputError):
                limits.public_target(url)

    def test_optional_host_allowlist(self):
        with patch.dict("os.environ", {"IMAGE_URL_ALLOWED_HOSTS": "res.cloudinary.com"}):
            with self.assertRaises(limits.ImageInputError):
                limits.public_target("https://example.com/a.jpg")

    def test_https_connection_is_pinned_and_hostname_verification_is_kept(self):
        data = cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes()
        pool = Mock()
        pool.urlopen.return_value = Response(data)
        with patch.object(limits.socket, "getaddrinfo", side_effect=[address("8.8.8.8"), address("127.0.0.1")]) as dns, \
                patch.object(limits.urllib3, "HTTPSConnectionPool", return_value=pool) as factory:
            result = limits.download_image("https://example.com/a.jpg")
        self.assertEqual(result.shape, (40, 40, 3))
        self.assertEqual(dns.call_count, 1)
        self.assertEqual(factory.call_args.kwargs["host"], "8.8.8.8")
        self.assertEqual(factory.call_args.kwargs["assert_hostname"], "example.com")
        self.assertEqual(factory.call_args.kwargs["server_hostname"], "example.com")
        self.assertEqual(factory.call_args.kwargs["cert_reqs"], "CERT_REQUIRED")
        self.assertFalse(pool.urlopen.call_args.kwargs["redirect"])

    def test_redirect_to_private_target_blocked(self):
        pool = Mock()
        pool.urlopen.return_value = Response(status=302, headers={"Location": "http://127.0.0.1/private"})
        with patch.object(limits.socket, "getaddrinfo", side_effect=[address("8.8.8.8"), address("127.0.0.1")]), \
                patch.object(limits.urllib3, "HTTPSConnectionPool", return_value=pool):
            with self.assertRaises(limits.ImageInputError):
                limits.download_image("https://example.com/a.jpg")
        self.assertEqual(pool.urlopen.call_count, 1)

    def test_oversized_download_with_or_without_content_length(self):
        for headers in ({}, {"Content-Length": "1000"}):
            pool = Mock()
            pool.urlopen.return_value = Response(b"x" * 1000, headers=headers)
            with self.subTest(headers=headers), patch.object(limits, "MAX_IMAGE_BYTES", 100), \
                    patch.object(limits.socket, "getaddrinfo", return_value=address("8.8.8.8")), \
                    patch.object(limits.urllib3, "HTTPSConnectionPool", return_value=pool):
                with self.assertRaises(limits.ImageInputError) as error:
                    limits.download_image("https://example.com/a.jpg")
                self.assertEqual(error.exception.status_code, 413)

    def test_redirect_loop_is_bounded(self):
        pool = Mock()
        pool.urlopen.side_effect = lambda *a, **kw: Response(status=302, headers={"Location": "/again"})
        with patch.object(limits.socket, "getaddrinfo", return_value=address("8.8.8.8")), \
                patch.object(limits.urllib3, "HTTPSConnectionPool", return_value=pool):
            with self.assertRaises(limits.ImageInputError):
                limits.download_image("https://example.com/a.jpg")
        self.assertEqual(pool.urlopen.call_count, limits.MAX_URL_REDIRECTS + 1)
