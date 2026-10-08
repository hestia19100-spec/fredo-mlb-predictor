"""Offline tests for the bounded direct historical NHL final transport."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.nhl.direct_final_transport import NHLDirectHTTPError, curl_direct_nhl_json
from src.nhl.official_final_capture import MAX_BYTES

URL = "https://api-web.nhle.com/v1/gamecenter/2021020006/landing"


class DirectFinalTransportTests(TestCase):
    def test_one_bounded_direct_request(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            Path(args[args.index("--output") + 1]).write_bytes(b'{"id":2021020006}')
            return subprocess.CompletedProcess(
                args, 0, ("200\n" + URL + "\napplication/json; charset=utf-8\n").encode(), b"",
            )

        with patch("src.nhl.direct_final_transport.shutil.which", return_value="/usr/bin/curl"), patch(
            "src.nhl.direct_final_transport.subprocess.run", side_effect=fake_run
        ):
            with curl_direct_nhl_json(URL, 15) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.geturl(), URL)
                self.assertEqual(response.read(100), b'{"id":2021020006}')
        args, kwargs = calls[0]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--location", args)
        self.assertNotIn("--user-agent", args)
        self.assertEqual(args[args.index("--proto") + 1], "=https")
        self.assertEqual(args[args.index("--max-filesize") + 1], str(MAX_BYTES + 1))
        self.assertEqual(args[args.index("--max-redirs") + 1], "0")
        self.assertEqual(kwargs["timeout"], 17)

    def test_rejects_out_of_scope_before_network(self):
        for url in ("http://api-web.nhle.com/v1/gamecenter/2021020006/landing",
                    URL + "?x=1", "https://other.example/v1/gamecenter/2021020006/landing"):
            with self.subTest(url=url), self.assertRaises(NHLDirectHTTPError):
                curl_direct_nhl_json(url, 15)
        with self.assertRaises(NHLDirectHTTPError):
            curl_direct_nhl_json(URL, 16)
        with patch("src.nhl.direct_final_transport.shutil.which", return_value=None):
            with self.assertRaises(NHLDirectHTTPError):
                curl_direct_nhl_json(URL, 15)

    def test_rejects_bad_http_metadata_and_oversize(self):
        cases = (
            ("302", URL, "application/json", b"{}", 0),
            ("200", URL + "/redirect", "application/json", b"{}", 0),
            ("200", URL, "text/html", b"{}", 0),
            ("403", URL, "application/json", b"{}", 22),
            ("200", URL, "application/json", b"x" * (MAX_BYTES + 1), 0),
            ("200", URL, "application/json", b"", 0),
        )
        for status, effective_url, content_type, body, code in cases:
            with self.subTest(status=status, effective_url=effective_url, content_type=content_type, length=len(body)):
                def fake_run(args, **_kwargs):
                    Path(args[args.index("--output") + 1]).write_bytes(body)
                    output = f"{status}\n{effective_url}\n{content_type}\n".encode()
                    return subprocess.CompletedProcess(args, code, output, b"")
                with patch("src.nhl.direct_final_transport.shutil.which", return_value="/usr/bin/curl"), patch(
                    "src.nhl.direct_final_transport.subprocess.run", side_effect=fake_run
                ):
                    with self.assertRaises(NHLDirectHTTPError):
                        curl_direct_nhl_json(URL, 15)

    def test_rejects_timeout(self):
        with patch("src.nhl.direct_final_transport.shutil.which", return_value="/usr/bin/curl"), patch(
            "src.nhl.direct_final_transport.subprocess.run",
            side_effect=subprocess.TimeoutExpired("curl", 17),
        ):
            with self.assertRaises(NHLDirectHTTPError):
                curl_direct_nhl_json(URL, 15)
