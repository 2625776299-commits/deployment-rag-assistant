"""Current-build A/B core evaluation. Offline preflight unless --live is explicit.

This isolates question/evidence processing with an empty shared conversation
context. It reuses the current contextual evaluator/controller and answer prompt;
task extraction, persistence and task switching belong to the separate HTTP suite.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from app.answering import validate_answer
from app.build_runtime import BuildManifest, build_request_binder, load_build
from app.conversation_context import ContextState
from app.conversation_rag import (build_conversation_answer_messages,
    build_initial_query, run_contextual_rag)
from app.embedding_runtime import EmbeddingConfig, create_embedding_client, embed_texts
from app.requirement_evaluation import build_requirement_messages
from app.retrieval_contract import RetrievalRequest

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_rrf_direct', 'B_coverage_flow')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def empty_context():
    return ContextState(input_version=1, current_facts=(), changes=(), pending_facts=())


def direct_answer(question, evidence, raw):
    """Same prompt builder, citation normalization and one format repair as B."""
    messages = build_conversation_answer_messages(question, evidence, empty_context())
    for attempt in range(2):
        text = raw('answer', messages)
        try:
            return validate_answer(text, evidence, normalize_citation_format=True)
        except ValueError as error:
            if attempt or not isinstance(text, str):
                raise
            details = {'validation_error': str(error),
                'allowed_citation_ids': [e['chunk_id'] for e in evidence]}
            messages += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content':
                '上次输出未通过格式/引用校验：' + json.dumps(details, ensure_ascii=False)
                + '。请重新输出完整JSON对象，必须包含status、answer、citation_ids、missing_information。'
                  '引用只能逐字使用上述ID，正文方括号引用与citation_ids一致；不能省略字段或使用中文标点替代JSON语法。'
                  '保持原问题、本轮分析范围、环境和同一份证据；不要新增现场事实、资料或为通过检查强改为answer。'}]


def execute_arm(question, initial, *, arm, pinned, raw):
    started = perf_counter()
    try:
        if arm == ARMS[0]:
            selected = deepcopy(initial[:3])
            if sum(len(c['content']) for c in selected) > 3600:
                raise ValueError('baseline_exceeds_shared_evidence_budget')
            pinned.check_evidence(selected)
            answer = direct_answer(question, selected, raw)
            result = {'status': answer['status'], 'message': answer['answer'],
                'answer': answer, 'evidence': selected, 'additional_retrievals': 0}
        else:
            first = True
            def shared_search(request):
                nonlocal first
                if first:
                    first = False
                    if request.build_id != pinned.build.build_id or request.query != question:
                        raise ValueError('shared_initial_request_changed')
                    return deepcopy(initial)
                return pinned.search(request)
            result = run_contextual_rag(question, empty_context(), retrieve=pinned.retrieve,
                request_retrieval=shared_search, build_id=pinned.build.build_id,
                evaluate_raw=lambda messages: raw('evaluate', messages),
                generate_raw=lambda messages: raw('answer', messages))
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
    result.update(arm=arm, elapsed_seconds=round(perf_counter() - started, 6),
                  semantic_review='pending', initial_candidates_reused=True)
    return result


def main():
    import os
    import sys
    from app.ask import create_generation_client, generation_options
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path,
        default=PROJECT / 'data/evaluations/current_version_20261105/single_turn_cases.json')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    suite_path = args.suite.resolve()
    suite = json.loads(suite_path.read_text(encoding='utf-8'))
    case_ids = args.cases or [c['case_id'] for c in suite['cases']]
    by_id = {c['case_id']: c for c in suite['cases']}
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(k not in by_id for k in case_ids):
        parser.error('Select existing unique case IDs')
    release_path = PROJECT / suite['release_path']
    if digest(release_path) != suite['release_sha256']:
        raise ValueError('frozen_release_changed')
    release = json.loads(release_path.read_text(encoding='utf-8'))
    manifest = BuildManifest(**release['candidate'])
    if manifest.to_dict() != suite['build']:
        raise ValueError('suite_build_identity_changed')
    root = release_path.parent / release['artifact_root']
    chunks, _ = load_build(manifest, root)
    if (os.getenv('GENERATION_MODEL'), os.getenv('GENERATION_BASE_URL', '').rstrip('/')) != (
            suite['generation_model'], suite['generation_base_url']):
        raise ValueError('generation_configuration_differs_from_frozen_suite')
    config = EmbeddingConfig.from_build(manifest)
    context = empty_context()
    for key in case_ids:
        question = by_id[key]['question']
        assert build_initial_query(question, context) == question
        build_conversation_answer_messages(question, chunks[:3], context)
        build_requirement_messages(question, chunks[:5], context=context, include_user_reply=True)
    base = PROJECT / 'data/current_evaluations'
    base.mkdir(exist_ok=True)
    output = Path(mkdtemp(prefix=('live_' if args.live else 'preflight_')
        + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=base))
    source_hashes = {str(p.relative_to(PROJECT)): digest(p) for p in sorted((PROJECT / 'app').glob('*.py'))}
    for protected in (release_path, root / manifest.artifact_directory / 'chunks.json',
                       root / manifest.artifact_directory / 'vectors.json'):
        source_hashes[str(protected.relative_to(PROJECT))] = digest(protected)
    def unchanged():
        if digest(suite_path) != suite_hash or any(digest(PROJECT / p) != h for p,h in source_hashes.items()):
            raise ValueError('evaluation_inputs_or_code_changed')
    suite_hash = digest(suite_path)
    settings = {'status': 'running' if args.live else 'offline_preflight_passed',
        'suite_sha256': suite_hash, 'release_sha256': digest(release_path),
        'source_sha256': source_hashes, 'case_ids': case_ids, 'build': manifest.to_dict(),
        'generation_model': suite['generation_model'], 'generation_base_url': suite['generation_base_url'],
        'scope': 'single_turn_contextual_core_without_task_preparation_or_HTTP',
        'shared_initial': True, 'sdk_retries': 0, 'daily_quota': None}
    write(output / 'configuration.json', settings)
    write(output / 'case_set.json', suite)
    if not args.live:
        write(output / 'summary.json', {'status': 'offline_preflight_passed', 'cases': len(case_ids),
            'planned_arms': 2 * len(case_ids), 'chunks': len(chunks), 'model_calls': 0})
        print(json.dumps({'status': 'offline_preflight_passed', 'cases': len(case_ids),
            'chunks': len(chunks), 'model_calls': 0, 'output': str(output)}, ensure_ascii=False))
        return
    events, results = [], []
    active = {'case_id': None, 'arm': None}
    def record(kind, stage, callback):
        event = {'attempt': len(events) + 1, 'kind': kind, 'stage': stage, **active, 'status': 'started'}
        events.append(event)
        write(output / 'attempts.json', events)
        begin = perf_counter()
        try:
            value = callback(event)
            event['status'] = 'returned'
            return value
        except Exception as error:
            event.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            event['seconds'] = round(perf_counter() - begin, 6)
            write(output / 'attempts.json', events)
    def raw(stage, messages):
        def call(event):
            prefix = str(event['attempt']) + '_' + stage
            write(output / (prefix + '_input.json'), messages)
            try:
                client, model = create_generation_client(model=suite['generation_model'])
                with client:
                    if str(client.base_url).rstrip('/') != suite['generation_base_url']:
                        raise ValueError('generation_endpoint_changed')
                    response = client.chat.completions.create(model=model, messages=messages,
                        **generation_options(client, evaluate=stage == 'evaluate', model=model))
            except Exception as error:
                write(output / (prefix + '_output.json'), {'error_type': type(error).__name__,
                    'http_status': getattr(error, 'status_code', None)})
                raise
            choice = response.choices[0] if response.choices else None
            write(output / (prefix + '_output.json'), {'requested_model': model,
                'returned_model': response.model, 'finish_reason': choice.finish_reason if choice else None,
                'raw_text': choice.message.content if choice else None,
                'usage': response.usage.model_dump() if response.usage else None})
            if choice is None or choice.finish_reason != 'stop':
                raise ValueError('model_response_incomplete')
            return choice.message.content
        return record('generation', stage, call)
    def bind_embedding(build):
        pinned_config = EmbeddingConfig.from_build(build)
        def query(text):
            usage = {}
            def call(event):
                try:
                    with create_embedding_client(config=pinned_config) as client:
                        return embed_texts(client, [text], config=pinned_config, usage_report=usage)[0]
                finally:
                    write(output / (str(event['attempt']) + '_embedding_usage.json'), usage)
            return record('embedding', 'query_embedding', call)
        return query
    class Catalog:
        def read_active(self): return manifest
    binder = build_request_binder(Catalog(), root, bind_embedding=bind_embedding,
        evaluate=lambda *args: None, generate=lambda *args: None)
    for index,key in enumerate(case_ids):
        unchanged()
        active.update(case_id=key, arm='shared_initial')
        question = by_id[key]['question']
        pinned = binder()
        begin = perf_counter()
        initial_error = None
        try:
            initial = pinned.search(RetrievalRequest(question, build_id=manifest.build_id))
        except Exception as error:
            initial, initial_error = [], type(error).__name__
        initial_seconds = round(perf_counter() - begin, 6)
        write(output / (key + '_initial.json'), {'question': question, 'candidates': initial,
            'seconds': initial_seconds, 'error_type': initial_error})
        for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS))):
            active['arm'] = arm
            before = len(events)
            result = ({'status': 'failed', 'error_type': initial_error} if initial_error else
                execute_arm(question, initial, arm=arm, pinned=pinned, raw=raw))
            result.update(case_id=key, shared_initial_seconds=initial_seconds,
                pipeline_seconds=round(initial_seconds + result['elapsed_seconds'],6)
                    if 'elapsed_seconds' in result else None,
                actual_model_attempts=len(events)-before)
            results.append(result)
            write(output / (key + '_' + arm + '.json'), result)
            write(output / (key + '_' + arm + '_review.json'), {
                'case_id': key, 'arm': arm, 'rubric': by_id[key]['review'],
                'evidence_coverage': None, 'answer_coverage': None,
                'semantic_pass': None, 'review_status': 'pending'})
            write(output / 'results.json', results)
            write(output / 'summary.json', {'status': 'running', 'planned_arms':2*len(case_ids),
                'completed_arms':len(results), 'failed_arms':sum(r['status']=='failed' for r in results),
                'generation_attempts':sum(e['kind']=='generation' for e in events),
                'embedding_attempts':sum(e['kind']=='embedding' for e in events),
                'semantic_accuracy':None})
            print(json.dumps({'case_id':key,'arm':arm,'status':result['status'],'output':str(output)}),flush=True)
        write(output / (key + '_retrieval_trace.json'), pinned.retriever.trace)
    unchanged()
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    summary.update(status='execution_complete_review_pending',source_unchanged=True)
    write(output/'summary.json',summary)


if __name__ == '__main__':
    main()
