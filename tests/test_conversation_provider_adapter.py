"""Real SDK exception types through the runtime adapter, without network/SQL."""
from pathlib import Path
import json
import socket
import ssl
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.conversation_runtime import create_conversation_learning_app
from app.provider_errors import ModelProviderError, provider_exception_diagnostics


class ConversationProviderAdapterTests(unittest.TestCase):
    def test_sdk_failures_are_classified_without_leaking_error_body(self):
        root = Path(__file__).resolve().parents[1] / 'data/architecture_demos'
        root.mkdir(parents=True, exist_ok=True)
        request = httpx.Request('POST', 'https://api.deepseek.com/v1/chat/completions')
        cases = (
            ('balance', APIStatusError('private provider detail',
                response=httpx.Response(402, request=request), body={'secret': 'private provider detail'}),
                ModelProviderError, 'provider_balance_insufficient'),
            ('connection', APIConnectionError(request=request), ModelProviderError, 'provider_connection_failed'),
            ('timeout', APITimeoutError(request=request), TimeoutError, 'model_timeout'),
        )
        with tempfile.TemporaryDirectory(prefix='provider_adapter_', dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root.resolve()))
            for name, upstream, expected_type, code in cases:
                with self.subTest(kind=name):
                    client = MagicMock()
                    client.__enter__.return_value = client
                    client.__exit__.return_value = False
                    client.base_url = 'https://api.deepseek.com/v1/'
                    client.chat.completions.create.side_effect = upstream
                    connect = Mock(side_effect=AssertionError('unexpected_SQL'))
                    output = Path(folder) / name
                    with patch('app.ask.create_generation_client', return_value=(client, 'deepseek-flash')), \
                         patch('app.conversation_runtime.PostgresConversationStore'):
                        app = create_conversation_learning_app(connect, Mock(), root, output,
                            schema='rag_conversation_test', authenticate=lambda _: 'U1')
                        with self.assertRaises(expected_type) as failed:
                            app.state.conversation_runtime['service'].generate_task([{'role': 'user', 'content': 'fixture'}])
                    self.assertEqual(str(failed.exception), code)
                    self.assertEqual(client.chat.completions.create.call_count, 1)
                    connect.assert_not_called()
                    log = (output / 'model_1_task_output.json').read_text(encoding='utf-8')
                    self.assertNotIn('private provider detail', log)
                    self.assertNotIn('secret', log)
                    detail = json.loads(log)
                    self.assertEqual(detail['diagnostics']['exception_chain'][0]['type'], type(upstream).__name__)

    def test_transport_causes_are_retained_without_messages_or_request_credentials(self):
        request = httpx.Request('POST', 'https://api.deepseek.com/v1/chat/completions',
                                headers={'Authorization': 'Bearer private-api-key'})
        tls = ssl.SSLEOFError(8, 'private-api-key private provider detail')
        connection = httpx.ConnectError('private-api-key', request=request)
        connection.__cause__ = tls
        upstream = APIConnectionError(message='private provider detail', request=request)
        upstream.__cause__ = connection
        details = provider_exception_diagnostics(upstream)
        self.assertEqual([x['type'] for x in details['exception_chain']],
                         ['APIConnectionError', 'ConnectError', 'SSLEOFError'])
        self.assertEqual(details['exception_chain'][-1]['errno'], 8)
        self.assertEqual(details['layer_hints'], ['tls', 'transport_connection'])
        serialized = json.dumps(details)
        self.assertNotIn('private-api-key', serialized)
        self.assertNotIn('private provider detail', serialized)
        self.assertNotIn('Authorization', serialized)

    def test_proxy_dns_context_and_cyclic_chains_are_bounded(self):
        dns = socket.gaierror(11001, 'private hostname')
        proxy = httpx.ProxyError('private proxy credentials')
        proxy.__context__ = dns
        details = provider_exception_diagnostics(proxy)
        self.assertEqual(details['layer_hints'], ['dns', 'proxy'])
        self.assertEqual(details['exception_chain'][-1]['errno'], 11001)
        dns.__cause__ = proxy
        details = provider_exception_diagnostics(proxy)
        self.assertEqual(len(details['exception_chain']), 2)
        self.assertTrue(details['chain_truncated_or_cyclic'])
        proxy.__suppress_context__ = True
        self.assertEqual(len(provider_exception_diagnostics(proxy)['exception_chain']), 1)
