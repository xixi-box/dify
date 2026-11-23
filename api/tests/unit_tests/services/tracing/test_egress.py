import logging
import socket
import ssl
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import override

import httpcore
import httpx
import pytest
from dify_trace_sdk.egress import EgressResponse
from dify_trace_sdk.errors import EgressBlockedError, EgressError

from app_factory import create_flask_app_with_configs
from core.helper import http_client_pooling, ssrf_proxy
from core.tools.errors import ToolSSRFError
from services.tracing.egress import MAX_RESPONSE_BYTES, SSRFEgress
from tests.helpers.unit.config_override import apply_config_overrides


class _ProxyStream(httpcore.MockStream):
    def __init__(self, tls_error: Exception | None) -> None:
        super().__init__(
            [b"HTTP/1.1 200 Connection Established\r\n\r\n", b"HTTP/1.1 202 Accepted\r\nContent-Length: 2\r\n\r\nOK"]
        )
        self.tls_error = tls_error
        self.closed = False
        self.writes: list[bytes] = []

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        super().write(buffer, timeout)
        self.writes.append(buffer)

    @override
    def close(self) -> None:
        self.closed = True
        super().close()

    @override
    def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.NetworkStream:
        assert server_hostname == "collector.example.com"
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert timeout == 0.1
        if self.tls_error is not None:
            raise self.tls_error
        return self


class _ProxyBackend(httpcore.NetworkBackend):
    def __init__(self, streams: Iterable[_ProxyStream]) -> None:
        self.streams = iter(streams)

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        assert (host, port) == ("proxy.example.com", 8080)
        assert timeout == 0.1
        assert local_address is None
        assert socket_options is None
        return next(self.streams)


@pytest.mark.parametrize(
    ("tls_error", "error_type"),
    [
        pytest.param(
            httpcore.ConnectError(ssl.SSLCertVerificationError("synthetic-private-detail")),
            "ConnectError",
            id="certificate-rejected",
        ),
        pytest.param(httpcore.ConnectTimeout("synthetic-private-detail"), "ConnectTimeout", id="handshake-timeout"),
    ],
)
@pytest.mark.parametrize("sensitive_url", [False, True], ids=["public-url", "sensitive-url"])
def test_proxy_tunnel_releases_slot_for_next_trace_export(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tls_error: Exception,
    error_type: str,
    sensitive_url: bool,
) -> None:
    """A failed TLS upgrade frees the only tunnel slot before the next export."""
    apply_config_overrides(
        monkeypatch, SSRF_PROXY_ALL_URL="http://proxy.example.com:8080", HTTP_REQUEST_NODE_SSL_VERIFY=True
    )
    monkeypatch.setattr(ssrf_proxy, "_SSRF_CLIENT_LIMITS", httpx.Limits(max_connections=1))
    factory = http_client_pooling.HttpClientPoolFactory()
    monkeypatch.setattr(http_client_pooling, "_factory", factory)
    create_flask_app_with_configs()
    if sensitive_url:
        caplog.set_level(logging.DEBUG, logger="httpcore")
    failed_stream, healthy_stream = _ProxyStream(tls_error), _ProxyStream(None)

    url = "https://collector.example.com/synthetic-private-path"
    with ssrf_proxy._get_ssrf_client(True) as client:
        transport = client._transport_for_url(httpx.URL(url))
        assert isinstance(transport, httpx.HTTPTransport)
        pool = transport._pool
        assert isinstance(pool, httpcore.HTTPProxy)
        pool._network_backend = _ProxyBackend([failed_stream, healthy_stream])
        egress = SSRFEgress()

        def export() -> EgressResponse:
            return egress.request(
                "POST",
                url,
                headers={"Authorization": "Bearer synthetic-private-header"},
                content=b"synthetic-private-body",
                timeout=0.1,
                sensitive_url=sensitive_url,
            )

        with pytest.raises(EgressError, match="Destination unreachable") as raised:
            export()

        assert raised.value.request_sent is False
        assert raised.value.__context__ is None
        assert pool.connections == [], "the failed CONNECT tunnel still occupies the only pool slot"
        assert failed_stream.closed
        assert b"CONNECT collector.example.com:443" in b"".join(failed_stream.writes)
        assert b"POST" not in b"".join(failed_stream.writes)
        assert error_type in caplog.text
        assert "synthetic-private" not in caplog.text

        response = export()
        assert (response.status_code, response.content) == (202, b"OK")
        assert b"POST /synthetic-private-path" in b"".join(healthy_stream.writes)


@dataclass(frozen=True)
class _ProxyCall:
    max_retries: int
    stream_response: bool
    timeout: float
    follow_redirects: bool
    headers: dict[str, str]


class _ProxyCalls:
    """Stands in for the SSRF proxy owner's ``make_request``."""

    def __init__(self, outcome: httpx.Response | Exception) -> None:
        self.outcome = outcome
        self.calls: list[_ProxyCall] = []

    def __call__(
        self,
        method: str,
        url: str,
        *,
        max_retries: int,
        stream_response: bool,
        timeout: float,
        follow_redirects: bool,
        headers: dict[str, str],
        content: bytes | None,
    ) -> httpx.Response:
        del method, url, content
        self.calls.append(
            _ProxyCall(
                max_retries=max_retries,
                stream_response=stream_response,
                timeout=timeout,
                follow_redirects=follow_redirects,
                headers=headers,
            )
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _request(egress: SSRFEgress, headers: dict[str, str]) -> None:
    egress.request(
        "POST",
        "https://collector.example.com/v1/traces",
        headers=headers,
        content=b"{}",
        timeout=5.0,
        sensitive_url=False,
    )


def test_requests_go_through_the_proxy_without_retries_redirects_or_ambient_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _ProxyCalls(httpx.Response(202, headers={"Content-Type": "application/json"}, content=b'{"ok": true}'))
    monkeypatch.setattr(ssrf_proxy, "make_request", proxy)

    result = SSRFEgress().request(
        "POST",
        "https://collector.example.com/v1/traces",
        headers={"Authorization": "Bearer token"},
        content=b"{}",
        timeout=5.0,
        sensitive_url=False,
    )

    assert (result.status_code, result.json(), result.headers["content-type"]) == (
        202,
        {"ok": True},
        "application/json",
    )
    (call,) = proxy.calls
    assert (call.max_retries, call.follow_redirects, call.stream_response, call.timeout) == (0, False, True, 5.0)
    assert call.headers["Accept-Encoding"] == "identity"
    assert call.headers["Cookie"] == ""
    assert call.headers["Authorization"] == "Bearer token"


def test_sdk_encodings_are_replaced_but_provider_cookies_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy = _ProxyCalls(httpx.Response(200, content=b""))
    monkeypatch.setattr(ssrf_proxy, "make_request", proxy)

    _request(SSRFEgress(), {"accept-encoding": "gzip, deflate, br", "cookie": "session=mine"})

    headers = proxy.calls[0].headers
    assert "accept-encoding" not in headers
    assert headers["Accept-Encoding"] == "identity"
    assert "Cookie" not in headers
    assert headers["cookie"] == "session=mine"


@pytest.mark.parametrize(
    ("failure", "blocked"),
    [(httpx.ProxyError("403 Forbidden"), True), (httpx.ProxyError("502 Bad Gateway"), False)],
)
def test_proxy_refusals_are_never_sent(monkeypatch: pytest.MonkeyPatch, failure: Exception, blocked: bool) -> None:
    monkeypatch.setattr(ssrf_proxy, "make_request", _ProxyCalls(failure))

    with pytest.raises(EgressError) as raised:
        _request(SSRFEgress(), {})

    assert isinstance(raised.value, EgressBlockedError) is blocked
    assert raised.value.request_sent is False


@pytest.mark.parametrize(
    ("failure", "expected", "request_sent"),
    [
        (ToolSSRFError("blocked"), EgressBlockedError, False),
        (httpx.ConnectError("refused"), EgressError, False),
        (httpx.ConnectTimeout("slow dns"), EgressError, False),
        (httpx.PoolTimeout("pool exhausted"), EgressError, False),
        (httpx.ReadTimeout("slow body"), EgressError, True),
        (httpx.RemoteProtocolError("reset"), EgressError, True),
    ],
)
def test_transport_failures_tell_whether_the_request_was_sent(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: type[EgressError], request_sent: bool
) -> None:
    monkeypatch.setattr(ssrf_proxy, "make_request", _ProxyCalls(failure))

    with pytest.raises(expected) as raised:
        _request(SSRFEgress(), {})

    assert raised.value.request_sent is request_sent
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_oversized_responses_count_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ssrf_proxy, "make_request", _ProxyCalls(httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1)))
    )

    with pytest.raises(EgressError) as raised:
        _request(SSRFEgress(), {})

    assert raised.value.request_sent is True


def _resolve_to(address: str) -> Callable[..., list[tuple[object, ...]]]:
    def resolve(host: str, port: int, type: int = 0) -> list[tuple[object, ...]]:
        del host
        return [(socket.AF_INET6 if ":" in address else socket.AF_INET, type, 6, "", (address, port))]

    return resolve


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "0.0.0.0", "::1", "::ffff:127.0.0.1"])
def test_grpc_channels_refuse_non_routable_destinations(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_to(address))

    with pytest.raises(EgressBlockedError):
        SSRFEgress().grpc_channel("https://collector.example.com:4317")


def test_grpc_channels_allow_private_collectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_to("10.0.0.8"))

    channel = SSRFEgress().grpc_channel("http://collector.internal:4317")

    channel.close()


@pytest.mark.parametrize(
    "endpoint", ["ftp://collector.example.com", "https://user:pass@collector.example.com", "https://"]
)
def test_grpc_channels_reject_malformed_endpoints(endpoint: str) -> None:
    with pytest.raises(EgressBlockedError):
        SSRFEgress().grpc_channel(endpoint)


def test_unresolvable_grpc_hosts_are_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(host: str, port: int, type: int = 0) -> list[tuple[object, ...]]:
        del host, port, type
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)

    with pytest.raises(EgressError) as raised:
        SSRFEgress().grpc_channel("https://collector.example.com")

    assert not isinstance(raised.value, EgressBlockedError)
    assert raised.value.request_sent is False
