"""Safe typed provider failures, independent of prompts, SDKs and credentials."""
PUBLIC_ERRORS = {
    'provider_balance_insufficient': ('本次模型请求因服务额度不足未能完成；请联系服务维护者确认额度。', False),
    'provider_connection_failed': ('本次请求未能连接模型服务，请稍后重试。', True),
}


class ModelProviderError(RuntimeError):
    def __init__(self, code, *, upstream_status=None):
        if code not in PUBLIC_ERRORS:
            raise ValueError('unknown_provider_error_code')
        super().__init__(code)
        self.code, self.upstream_status = code, upstream_status

    def detail(self, stage):
        return {'stage': stage, 'code': self.code, 'retryable': PUBLIC_ERRORS[self.code][1],
                'upstream_status': self.upstream_status}


def provider_public_reply(error):
    code = error.get('code')
    if code not in PUBLIC_ERRORS:
        return None
    message, retryable = PUBLIC_ERRORS[code]
    return {'status': 'failed', 'message': message, 'error_code': code, 'retryable': retryable}


def provider_exception_diagnostics(error):
    """Bounded internal type/code chain; never stringify provider exceptions.

    These hints identify reported transport layers, not the responsible route,
    node or service. SDK messages/bodies/requests may contain private data.
    """
    import re
    import socket
    import ssl
    import httpx

    chain, hints, seen = [], set(), set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        item = {'type': type(current).__name__, 'module': type(current).__module__}
        for name in ('errno', 'winerror', 'verify_code'):
            value = getattr(current, name, None)
            if type(value) is int:
                item[name] = value
        if isinstance(current, ssl.SSLError):
            hints.add('tls')
            for name in ('reason', 'library'):
                value = getattr(current, name, None)
                if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_]{1,100}', value):
                    item['ssl_' + name] = value
        for kind, hint in ((socket.gaierror, 'dns'),
                           (httpx.ProxyError, 'proxy'),
                           (httpx.ConnectError, 'transport_connection'),
                           (httpx.ReadError, 'transport_read'),
                           (httpx.WriteError, 'transport_write'),
                           (httpx.TimeoutException, 'timeout')):
            if isinstance(current, kind):
                hints.add(hint)
        chain.append(item)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return {'exception_chain': chain, 'layer_hints': sorted(hints),
            'chain_truncated_or_cyclic': current is not None, 'messages_logged': False}
