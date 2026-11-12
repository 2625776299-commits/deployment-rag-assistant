"""Explicit runnable multi-turn profile: PostgreSQL + pinned corpus + live providers."""
import argparse
from datetime import date
import json
import os
from pathlib import Path

from app.build_runtime import BuildManifest, build_request_binder, load_build
from app.conversation_context import restore_context
from app.conversation_rag import run_prepared_snapshot
from app.conversation_service import ConversationExecution, ConversationService
from app.conversation_store import PostgresConversationStore
from app.main import create_app
from app.provider_errors import ModelProviderError, provider_exception_diagnostics


def create_conversation_learning_app(connect, catalog, artifact_root, output, *, schema, authenticate):
    from app.ask import create_generation_client, generation_options
    from app.embedding_runtime import EmbeddingConfig, create_embedding_client, embed_texts
    from app.learning_runtime import RunAccounting
    from openai import APIStatusError, APITimeoutError, APIConnectionError
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    accounting = RunAccounting(output / 'attempts.json', date.today().isoformat())
    store = PostgresConversationStore(connect, schema=schema)
    store.initialize()
    initial_client, model = create_generation_client()
    with initial_client:
        base_url = str(initial_client.base_url)

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def raw(stage):
        def generate(messages):
            attempt = accounting.reserve('generation', stage)
            write(output / f'model_{attempt}_{stage}_input.json', messages)
            path = output / f'model_{attempt}_{stage}_output.json'
            active, requested = create_generation_client(model=model)
            with active:
                if str(active.base_url) != base_url or requested != model:
                    raise ValueError('generation_configuration_changed')
                try:
                    response = active.chat.completions.create(model=model, messages=messages,
                        **generation_options(active, evaluate=stage == 'evaluate', model=model))
                except Exception as error:
                    detail = {'requested_model': model, 'error_type': type(error).__name__,
                              'diagnostics': provider_exception_diagnostics(error)}
                    if isinstance(error, APIStatusError):
                        detail['http_status'] = error.status_code
                    write(path, detail)
                    if isinstance(error, APITimeoutError):
                        raise TimeoutError('model_timeout') from None
                    if isinstance(error, APIStatusError) and error.status_code == 402:
                        raise ModelProviderError('provider_balance_insufficient', upstream_status=402) from None
                    if isinstance(error, APIConnectionError):
                        raise ModelProviderError('provider_connection_failed') from None
                    raise RuntimeError('model_request_failed') from None
            choice = response.choices[0] if response.choices else None
            write(path, {'requested_model': model, 'returned_model': response.model,
                'finish_reason': choice.finish_reason if choice else None,
                'raw_text': choice.message.content if choice else None,
                'usage': response.usage.model_dump() if response.usage else None})
            if choice is None or choice.finish_reason != 'stop':
                raise ValueError('model_response_incomplete')
            return choice.message.content
        return generate

    def bind_embedding(build):
        config = EmbeddingConfig.from_build(build)
        def query(text):
            attempt = accounting.reserve('embedding', 'query_embedding')
            usage = {}
            try:
                with create_embedding_client(config=config) as client:
                    return embed_texts(client, [text], usage_report=usage, config=config)[0]
            finally:
                write(output / f'embedding_{attempt}_usage.json', usage)
        return query

    def unused(*args):
        raise AssertionError('conversation_uses_context_aware_raw_callbacks')
    factory = build_request_binder(catalog, Path(artifact_root), bind_embedding=bind_embedding,
                                   evaluate=unused, generate=unused)
    bindings = []
    def bind(build_id=None):
        pinned = factory(build_id)
        bindings.append(pinned)
        def execute(snapshot):
            # Scope has already been projected by the authorized store snapshot.
            restore_context(snapshot['context'])
            return run_prepared_snapshot(snapshot, retrieve=pinned.retrieve,
                request_retrieval=pinned.search, build_id=pinned.build.build_id,
                evaluate_raw=raw('evaluate'), generate_raw=raw('answer'))
        return ConversationExecution(pinned.build.build_id, execute)
    service = ConversationService(store, generate_task=raw('task'), generate_context=raw('context'), include_constraints=True,
                                  bind_execution=bind, resume_execution=bind)
    app = create_app(conversation_service=service, authenticate=authenticate)
    app.state.conversation_runtime = {'store': store, 'service': service, 'bindings': bindings,
        'accounting': accounting, 'output': output, 'model': model}
    return app


def register_release(catalog, release_path):
    """Install this explicit release into a new catalogue; preserve an existing active build."""
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text(encoding='utf-8'))
    build = BuildManifest(**release['candidate'])
    root = release_path.parent / release['artifact_root']
    load_build(build, root)
    catalog.initialize()
    try:
        active = catalog.read_active()
    except ValueError as error:
        if str(error) != 'no_ready_active_build':
            raise
        active = None
    if active is not None:
        if active != build:
            raise ValueError('existing_catalog_has_different_active_release')
        return root
    catalog.register(build)
    catalog.validate_ready(build.build_id, root)
    catalog.publish(build.build_id, expected_active=None)
    return root


def main():
    from dotenv import load_dotenv
    import psycopg
    import uvicorn
    from app.build_catalog import PostgresBuildCatalog
    from app.conversation_api import bearer_authenticator
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    configured_release = os.getenv('RAG_CONVERSATION_RELEASE_PATH', 'data/corpus_releases/deploy_faults_20261102_v1/release.json')
    parser.add_argument('--release', type=Path, default=project / configured_release)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    dsn = os.getenv('RAG_CONVERSATION_DATABASE_URL', '')
    token = os.getenv('RAG_CONVERSATION_TOKEN', '')
    if not dsn or not token:
        raise SystemExit('Set RAG_CONVERSATION_DATABASE_URL and RAG_CONVERSATION_TOKEN locally; do not paste secrets into chat.')
    def connect():
        return psycopg.connect(dsn, connect_timeout=5)
    catalog = PostgresBuildCatalog(connect, schema='rag_build_conversation_learning')
    root = register_release(catalog, args.release)
    from tempfile import mkdtemp
    demo_root = project / 'data/architecture_demos'
    demo_root.mkdir(parents=True, exist_ok=True)
    output = args.output or Path(mkdtemp(prefix='conversation_http_', dir=demo_root)) / 'runtime'
    app = create_conversation_learning_app(connect, catalog, root, output,
        schema='rag_conversation_learning', authenticate=bearer_authenticator({token: 'learning_user'}))
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
