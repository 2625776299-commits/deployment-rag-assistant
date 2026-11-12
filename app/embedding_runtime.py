"""Active embedding adapter. Historical embeddings.py stays frozen for old runs."""

import math
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError
from app.loader import PROJECT_ROOT
from app import embeddings as legacy

load_dotenv(PROJECT_ROOT / '.env', override=False)
MODEL = os.getenv('EMBEDDING_MODEL', legacy.MODEL)
BASE_URL = os.getenv('EMBEDDING_BASE_URL', legacy.BASE_URL).rstrip('/')
PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'legacy')
DIMENSIONS = (int(os.getenv('EMBEDDING_DIMENSIONS',
    '3072' if MODEL == 'text-embedding-3-large' else '1536' if PROVIDER == 'openai' else '1024'))
    if PROVIDER in ('dashscope', 'openai') else None)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    dimensions: int | None

    @classmethod
    def from_build(cls, build):
        endpoint = build.embedding_base_url.rstrip('/')
        host = urlsplit(endpoint).hostname or ''
        if endpoint == 'https://api.openai.com/v1':
            provider = 'openai'
        elif host == 'dashscope.aliyuncs.com' or host.endswith('.cn-beijing.maas.aliyuncs.com'):
            provider = 'dashscope'
        else:
            raise ValueError('Unsupported pinned embedding endpoint')
        return cls(provider, build.embedding_model, endpoint, build.embedding_dimension)


def _config(config):
    return config or EmbeddingConfig(PROVIDER, MODEL, BASE_URL, DIMENSIONS)


def create_embedding_client(*, config=None):
    config = _config(config)
    if config.provider not in ('dashscope', 'openai'):
        return legacy.create_embedding_client()
    parsed = urlsplit(config.base_url)
    if config.provider == 'openai':
        if config.base_url.rstrip('/') != 'https://api.openai.com/v1':
            raise ValueError('OpenAI embedding key requires the official HTTPS endpoint')
        key = os.getenv('EMBEDDING_API_KEY', '').strip()
        if not key and os.getenv('GENERATION_BASE_URL', '').rstrip('/') == config.base_url.rstrip('/'):
            key = os.getenv('GENERATION_API_KEY', '').strip()
        if not key:
            raise ValueError('Set EMBEDDING_API_KEY or the same-endpoint GENERATION_API_KEY locally')
        return OpenAI(api_key=key, base_url=config.base_url, timeout=30, max_retries=0)
    if parsed.scheme != 'https' or not (parsed.hostname == 'dashscope.aliyuncs.com'
            or (parsed.hostname or '').endswith('.cn-beijing.maas.aliyuncs.com')):
        raise ValueError('DashScope key requires the configured official Beijing HTTPS endpoint')
    key = os.getenv('DASHSCOPE_API_KEY', '').strip()
    if not key:
        raise ValueError('Set DASHSCOPE_API_KEY locally')
    return OpenAI(api_key=key, base_url=config.base_url, timeout=30, max_retries=0)


def embed_texts(client, texts, *, usage_report=None, config=None):
    config = _config(config)
    model, dimension = config.model, config.dimensions
    if config.provider not in ('dashscope', 'openai'):
        return legacy.embed_texts(client, texts, usage_report=usage_report)
    if config.provider == 'dashscope' and (model != 'text-embedding-v4' or dimension not in (2048,1536,1024,768,512,256,128,64)):
        raise ValueError('Unsupported DashScope embedding configuration')
    native_dimensions = {'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072}
    if config.provider == 'openai' and (model not in native_dimensions
            or type(dimension) is not int or not 1 <= dimension <= native_dimensions[model]):
        raise ValueError('Unsupported OpenAI embedding configuration')
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError('Embedding inputs must be non-empty strings')
    report = usage_report if usage_report is not None else {}
    report.update(model=model, dimensions=dimension, text_count=len(texts),
                  total_tokens=0, requests=[])
    vectors = []
    batch_size = 10 if config.provider == 'dashscope' else 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        event = {'offset': start, 'count': len(batch), 'status': 'started'}
        report['requests'].append(event)
        try:
            response = client.embeddings.create(model=model, input=batch,
                dimensions=dimension, encoding_format='float')
            rows = sorted(response.data, key=lambda r: r.index)
            if response.model != model or [r.index for r in rows] != list(range(len(batch))):
                raise ValueError('Embedding model or input alignment differs')
            result = [r.embedding for r in rows]
            if any(len(v) != dimension or any(not math.isfinite(x) for x in v)
                   or not any(v) for v in result):
                raise ValueError('Invalid embedding dimension, values, or norm')
            tokens = response.usage.total_tokens if response.usage else None
            event.update(status='returned', total_tokens=tokens)
            report['total_tokens'] = (report['total_tokens'] + tokens
                if tokens is not None and report['total_tokens'] is not None else None)
            vectors.extend(result)
        except APIStatusError as ex:
            event.update(status='failed', http_status=ex.status_code)
            raise RuntimeError(f'Embedding API returned HTTP {ex.status_code}') from None
        except APIConnectionError:
            event.update(status='failed', error_type='APIConnectionError')
            raise RuntimeError('Could not reach embedding API') from None
        except Exception as ex:
            event.update(status='failed', error_type=type(ex).__name__)
            raise
    return vectors
