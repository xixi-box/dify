"""Temporary CONNECT tunnel cleanup for the httpcore 1.0.9 API lock."""

import logging
from functools import cache
from typing import override

import httpcore

logger = logging.getLogger(__name__)


@cache
def install_proxy_tunnel_cleanup() -> None:
    """Install cleanup for new CONNECT tunnels during API and worker startup."""
    # Remove with the 1.0.9 lock when a published release includes the fix.
    # https://github.com/encode/httpcore/pull/1087
    if httpcore.__version__ != "1.0.9":
        return

    from httpcore._async import http_proxy as async_http_proxy
    from httpcore._sync import http_proxy as sync_http_proxy
    from httpcore._synchronization import AsyncShieldCancellation

    class _TunnelHTTPConnection(sync_http_proxy.TunnelHTTPConnection):
        @override
        def handle_request(self, request: httpcore.Request) -> httpcore.Response:
            """Send a request and close an incomplete tunnel before propagating a setup failure."""
            try:
                return super().handle_request(request)
            except BaseException:
                if not self._connected and not self.is_closed():
                    try:
                        self.close()
                    except Exception as error:
                        logger.warning("Failed to close CONNECT tunnel (%s)", type(error).__name__)
                raise

    class _AsyncTunnelHTTPConnection(async_http_proxy.AsyncTunnelHTTPConnection):
        @override
        async def handle_async_request(self, request: httpcore.Request) -> httpcore.Response:
            """Send a request and close an incomplete tunnel even if setup was cancelled."""
            try:
                return await super().handle_async_request(request)
            except BaseException:
                if not self._connected and not self.is_closed():
                    with AsyncShieldCancellation():
                        try:
                            await self.aclose()
                        except Exception as error:
                            logger.warning("Failed to close CONNECT tunnel (%s)", type(error).__name__)
                raise

    # The pool looks these classes up when opening a tunnel. Before TLS
    # succeeds, the underlying HTTP/1.1 proxy connection is never available
    # for another request, so cleanup can run after the connect lock exits.
    sync_http_proxy.TunnelHTTPConnection = _TunnelHTTPConnection
    async_http_proxy.AsyncTunnelHTTPConnection = _AsyncTunnelHTTPConnection
