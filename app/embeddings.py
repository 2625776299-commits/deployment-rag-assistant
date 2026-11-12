import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from app.loader import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env", override=False)
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL", "https://api.openai-hk.com/v1"
).rstrip("/")


def create_embedding_client() -> OpenAI:
    """Use a dedicated key or reuse an existing key for the same endpoint."""
    api_key = os.getenv("EMBEDDING_API_KEY")
    existing_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if not api_key and existing_url == BASE_URL:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set EMBEDDING_API_KEY locally, or configure OPENAI_API_KEY and "
            "OPENAI_BASE_URL for the same embedding endpoint"
        )
    if urlparse(BASE_URL).scheme != "https":
        raise ValueError("Embedding endpoint must use HTTPS")
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=30.0, max_retries=0)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    usage_report: dict | None = None,
) -> list[list[float]]:
    """Convert each input string into one vector, preserving input order."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty strings")

    try:
        response = client.embeddings.create(
            model=MODEL,
            input=texts,
            encoding_format="float",
        )
    except APIStatusError as error:
        raise RuntimeError(
            f"Embedding API returned HTTP {error.status_code}; "
            "check provider access, model availability and account balance"
        ) from None
    except APIConnectionError:
        raise RuntimeError("Could not reach embedding API within the configured timeout") from None

    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(len(texts))):
        raise ValueError("Embedding response does not match the input batch")
    if response.model != MODEL:
        raise ValueError("Provider returned a different model than requested")
    vectors = [item.embedding for item in items]
    if not vectors[0] or any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("Embedding vectors must have the same non-zero dimension")

    tokens = response.usage.total_tokens if response.usage else "unknown"
    if usage_report is not None:
        usage_report.update({
            "model": response.model,
            "text_count": len(texts),
            "dimensions": len(vectors[0]),
            "total_tokens": response.usage.total_tokens if response.usage else None,
        })
    print(f"Embedding: {len(texts)} texts; {len(vectors[0])} dimensions; tokens: {tokens}")
    return vectors
