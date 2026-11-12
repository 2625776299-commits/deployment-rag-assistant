"""Minimal retrieve-then-answer CLI; generation settings must be explicit."""

import argparse
import json
import os
import sys
from urllib.parse import urlparse
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

from app.answering import build_messages, validate_answer
from app.embedding_runtime import BASE_URL as EMBEDDING_BASE_URL


def create_generation_client(model: str | None = None) -> tuple[OpenAI, str]:
    model = model or os.getenv("GENERATION_MODEL", "")
    base_url = os.getenv("GENERATION_BASE_URL", "").rstrip("/")
    if not model.strip():
        raise ValueError("Set GENERATION_MODEL or pass --model; a chat model is required")
    if not base_url or urlparse(base_url).scheme != "https":
        raise ValueError("Set GENERATION_BASE_URL to the provider's HTTPS API base URL")

    # Bind the new endpoint to its own key; never forward a previous provider's key.
    provider_host = urlparse(base_url).hostname
    api_key = (os.getenv("DEEPSEEK_API_KEY") if provider_host == "api.deepseek.com"
               else os.getenv("GENERATION_API_KEY"))
    if provider_host == 'api.openai.com' and not api_key:
        # Require the key deliberately supplied for this migration. A process-
        # level OPENAI_API_KEY may belong to a different account or legacy use.
        raise ValueError('Set GENERATION_API_KEY locally with your official OpenAI project key')
    if provider_host == 'api.deepseek.com' and not api_key:
        raise ValueError('Set DEEPSEEK_API_KEY locally')
    if not api_key and os.getenv("OPENAI_BASE_URL", "").rstrip("/") == base_url:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and EMBEDDING_BASE_URL == base_url:
        api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("Set GENERATION_API_KEY, or reuse a key configured for the same endpoint")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0), model


def generation_options(client, *, evaluate: bool = False, model: str | None = None) -> dict:
    provider_host = urlparse(str(client.base_url)).hostname
    if provider_host == 'api.openai.com':
        options = {'response_format': {'type': 'json_object'}}
        selected_model = model or os.getenv('GENERATION_MODEL', '')
        if selected_model == 'gpt-6.1-sol' or selected_model.startswith('gpt-6.1-sol-'):
            # Sol supports low..max, not DeepSeek's thinking switch or "none".
            options['reasoning_effort'] = 'high' if evaluate else 'low'
        return options
    if provider_host != 'api.deepseek.com':
        return {}
    # Evaluation has a nested evidence contract; let the model reason before JSON.
    options = {'extra_body': {'thinking': {'type': 'enabled' if evaluate else 'disabled'}},
               'response_format': {'type': 'json_object'}}
    if evaluate:
        options['reasoning_effort'] = 'high'
    return options


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    matches: list[dict],
    *,
    raw_output_path: Path | None = None,
) -> dict:
    messages = build_messages(question, matches)
    try:
        response = client.chat.completions.create(model=model, messages=messages,
                                                 **generation_options(client, model=model))
    except APIStatusError as error:
        raise RuntimeError(
            f"Generation API returned HTTP {error.status_code}; check model and provider access"
        ) from None
    except APIConnectionError:
        raise RuntimeError("Could not reach the generation API within the configured timeout") from None
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ValueError("Generation did not return a complete text answer")
    text = response.choices[0].message.content
    # Preserve failed outputs when a diagnostic path is explicitly requested.
    if raw_output_path is not None:
        raw_output_path.write_text(
            json.dumps({
                "requested_model": model,
                "returned_model": response.model,
                "raw_text": text,
                "usage": response.usage.model_dump() if response.usage else None,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    result = validate_answer(text, matches)
    result["requested_model"] = model
    result["returned_model"] = response.model
    usage = response.usage
    result["generation_usage"] = (
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage is not None else None
    )
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", help="Override GENERATION_MODEL")
    args = parser.parse_args()
    if not args.question.strip() or args.top_k <= 0:
        parser.error("question must be non-empty and top-k must be positive")
    # Only the historical search CLI needs Chroma. Importing the generation
    # adapter for the current pinned-corpus service must not load that stack.
    from app.search import retrieve_chunks
    # Validate generation configuration before spending a query embedding request.
    client, model = create_generation_client(args.model)
    with client:
        matches = retrieve_chunks(args.question, top_k=args.top_k)
        result = generate_answer(client, model, args.question, matches)
    result["question"] = args.question
    result["retrieved_chunks"] = [
        {"chunk_id": match["chunk_id"], "distance": match["distance"]}
        for match in matches
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
