"""Credential isolation and request compatibility for the official API switch."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.ask import create_generation_client, generation_options


class GenerationProviderConfigTests(unittest.TestCase):
    def test_official_endpoint_does_not_reuse_ambient_or_other_provider_keys(self):
        env = {
            'GENERATION_BASE_URL': 'https://api.openai.com/v1',
            'GENERATION_MODEL': 'gpt-6.1-sol',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'fixture-ambient-key',
            'DEEPSEEK_API_KEY': 'fixture-other-provider-key',
        }
        with patch.dict(os.environ, env, clear=True), patch('app.ask.OpenAI') as constructor:
            with self.assertRaisesRegex(ValueError, 'GENERATION_API_KEY'):
                create_generation_client()
            constructor.assert_not_called()

    def test_only_the_deliberately_configured_key_is_sent_to_each_endpoint(self):
        for host, selected_key in [('api.openai.com/v1', 'fixture-official-key'),
                                   ('api.deepseek.com', 'fixture-deepseek-key')]:
            with self.subTest(host=host):
                env = {'GENERATION_BASE_URL': 'https://' + host,
                       'GENERATION_MODEL': 'fixture-model',
                       'GENERATION_API_KEY': 'fixture-official-key',
                       'DEEPSEEK_API_KEY': 'fixture-deepseek-key',
                       'OPENAI_API_KEY': 'fixture-unrelated-key'}
                with patch.dict(os.environ, env, clear=True), patch('app.ask.OpenAI') as constructor:
                    create_generation_client()
                    self.assertEqual(constructor.call_args.kwargs['api_key'], selected_key)
                    self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)

    def test_json_contract_and_provider_specific_reasoning_parameters(self):
        official = SimpleNamespace(base_url='https://api.openai.com/v1/')
        deepseek = SimpleNamespace(base_url='https://api.deepseek.com/')
        for evaluate in (False, True):
            with self.subTest(evaluate=evaluate):
                options = generation_options(official, model='gpt-6.1-sol', evaluate=evaluate)
                self.assertEqual(options['response_format'], {'type': 'json_object'})
                self.assertEqual(options['reasoning_effort'], 'high' if evaluate else 'low')
                self.assertNotIn('extra_body', options)
                self.assertNotIn('temperature', options)
                original = generation_options(deepseek, model='deepseek-flash', evaluate=evaluate)
                self.assertEqual(original['extra_body']['thinking']['type'],
                                 'enabled' if evaluate else 'disabled')
        # Other models are not assumed to implement Sol's reasoning levels.
        self.assertNotIn('reasoning_effort', generation_options(official, model='gpt-4.1'))


if __name__ == '__main__':
    unittest.main()
