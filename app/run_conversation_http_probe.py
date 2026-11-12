"""Real localhost HTTP + isolated PostgreSQL + pinned live-provider development probe.

Existing corpus vectors are reused. No business database or existing service is
modified. The HTTP server and temporary PostgreSQL cluster stop in finally.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from secrets import token_urlsafe
import shutil
import socket
import subprocess
import sys
from tempfile import mkdtemp
from threading import Thread
from time import monotonic, sleep
from uuid import uuid4


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def load_reviewed_case(path, case_id):
    data = json.loads(path.read_text(encoding='utf-8'))
    selected = [c for c in data['cases'] if c['case_id'] == case_id]
    if len(selected) != 1 or selected[0]['review_status'] not in ('user_annotation_reviewed_by_assistant', 'assistant_reviewed'):
        raise ValueError('select_one_reviewed_development_case')
    case = selected[0]
    if not case_id.replace('_', '').isalnum():
        raise ValueError('invalid_case_id')
    turns = [(t['message_id'], t['content']) for t in case['turns']]
    if (not turns or len({mid for mid, _ in turns}) != len(turns)
            or any(not isinstance(v, str) or not v.strip() for pair in turns for v in pair)):
        raise ValueError('invalid_case_messages')
    # The HTTP payload receives ONLY these two strings. Rubrics and expected
    # actions stay in the experiment directory for review after execution.
    return data, case, turns


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pg-bin', type=Path)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--label', choices=('baseline', 'candidate'), default='baseline')
    args = parser.parse_args()
    if (args.suite is None) != (args.case_id is None):
        parser.error('--suite and --case-id must be supplied together')
    suite, selected, selected_turns = load_reviewed_case(args.suite, args.case_id) if args.suite else (None, None, None)
    project = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv
    load_dotenv(project / '.env', override=False)
    import httpx
    import psycopg
    import uvicorn
    from app.build_catalog import PostgresBuildCatalog
    from app.conversation_api import bearer_authenticator
    from app.conversation_runtime import create_conversation_learning_app, register_release
    from app.ask import create_generation_client
    from urllib.parse import urlparse
    client, model = create_generation_client()
    with client:
        expected_model = suite.get('generation_model') if suite else os.getenv('GENERATION_MODEL')
        if model != expected_model or urlparse(str(client.base_url)).scheme != 'https':
            raise ValueError('probe_generation_configuration_differs_from_suite')
    root = project / 'data/architecture_demos'
    root.mkdir(parents=True, exist_ok=True)
    prefix = 'conversation_http_' + ((args.case_id + '_' + args.label + '_') if selected else 'live_')
    output = Path(mkdtemp(prefix=prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=root))
    print('实验记录目录: ' + str(output), flush=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    suffix = '.exe' if os.name == 'nt' else ''
    pg_bin = args.pg_bin or Path(shutil.which('pg_ctl') or '').parent
    commands = {name: pg_bin / (name + suffix) for name in ('initdb', 'pg_ctl')}
    if any(not command.is_file() for command in commands.values()):
        raise ValueError('provide_postgresql_binaries')
    pg_port, http_port = free_port(), free_port()
    cluster = output / 'postgres'
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    connections, server, thread = [], None, None
    def connect():
        connection = psycopg.connect(host='127.0.0.1', port=pg_port, dbname='postgres',
                                    user='rag_conversation_probe', connect_timeout=5)
        connections.append(connection)
        return connection
    (output / 'code').mkdir()
    hashes = {}
    for name in ('main.py','conversation_api.py','conversation_service.py','conversation_runtime.py',
                 'conversation_store.py','conversation_preparation.py','task_proposal.py','context_proposal.py',
                  'conversation_rag.py','conversation_context.py','conversation_attempts.py','analysis_scope.py',
                  'analysis_preparation.py','claim_constraints.py','answering.py','build_catalog.py',
                 'build_runtime.py','provider_errors.py','rag_service.py','requirement_evaluation.py',
                  'run_conversation_http_probe.py','task_constraints.py','constraint_preparation.py',
                  'conversation_constraint_storage.py','question_authority.py','conversation_question_storage.py'):
        data = (project / 'app' / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (output / 'code' / name).write_bytes(data)
    report = {'scope': 'real localhost TCP HTTP; isolated PostgreSQL conversation/build catalogue; configured live providers; real hybrid retrieval/RAG',
              'status': 'started', 'semantic_review': 'pending', 'model': model,
              'source_sha256': hashes, 'turns': [], 'existing_business_data_touched': False}
    if selected:
        save('selected_case.json', selected)
        (output / 'frozen_suite.json').write_bytes(args.suite.read_bytes())
        report.update(case_id=args.case_id, comparison_role=args.label,
            case_review_status=selected['review_status'],
            suite_sha256=hashlib.sha256(args.suite.read_bytes()).hexdigest(), expected_fields_sent_to_model=False)
    try:
        with (output / 'startup.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(commands['initdb']), '-D', str(cluster), '-U', 'rag_conversation_probe',
                            '--auth=trust', '--no-locale', '--encoding=UTF8'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=45, creationflags=flags)
            subprocess.run([str(commands['pg_ctl']), '-D', str(cluster), '-l', str(output / 'postgres.log'),
                            '-o', f'-h 127.0.0.1 -p {pg_port}', '-w', '-t', '20', 'start'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=25, creationflags=flags)
        catalog = PostgresBuildCatalog(connect, schema='rag_build_http_' + uuid4().hex)
        release = args.release or project / os.getenv('RAG_CONVERSATION_RELEASE_PATH',
            'data/corpus_releases/deploy_faults_20261102_v1/release.json')
        artifact_root = register_release(catalog, release)
        token1, token2 = token_urlsafe(32), token_urlsafe(32)
        app = create_conversation_learning_app(connect, catalog, artifact_root, output / 'runtime',
            schema='rag_conversation_http_' + uuid4().hex,
            authenticate=bearer_authenticator({token1: 'U1', token2: 'U2'}))
        runtime = app.state.conversation_runtime
        store, accounting = runtime['store'], runtime['accounting']
        report.update(build_id=catalog.read_active().build_id, corpus_chunks=catalog.read_active().chunk_count,
                      embedding_model=catalog.read_active().embedding_model)
        if suite and report['build_id'] != suite['knowledge_build_id']:
            raise ValueError('suite_knowledge_build_mismatch')
        config = uvicorn.Config(app, host='127.0.0.1', port=http_port, log_level='warning', access_log=False)
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        deadline = monotonic() + 10
        while not server.started and monotonic() < deadline and thread.is_alive():
            sleep(0.02)
        if not server.started:
            raise RuntimeError('http_server_not_ready')
        cases = [
            ('M1', '我的应用连不上数据库，连接地址中的host和port应该怎么填？如果缺少关键环境信息，请先问我，不要猜具体地址。'),
            ('M2', '应用直接运行在宿主机上，数据库在同一台宿主机的Docker Compose容器中，服务名为db，发布端口映射是5433:5432。请按这些信息继续说明host和port。'),
            ('M3', '换一个独立任务：本地测试环境中，应用和数据库都在同一个Compose网络的两个容器里。只解释应用为什么可以用数据库服务名db访问它，不要沿用前面宿主机的端口配置。'),
            ('M4', '回到最开始的连接地址问题，继续按之前确认的宿主机部署说明host和port即可。'),
            ('M5', '补充前面那个数据库任务：应用的位置确认是在宿主机，但我没说清是最开始那个任务还是后面的本地测试任务，请先确认任务归属。'),
            ('M6', '指最开始那个填写host和port的任务，不是后面的本地测试任务。'),
        ]
        if selected_turns is not None:
            cases = selected_turns
        with httpx.Client(base_url=f'http://127.0.0.1:{http_port}', headers={'Authorization': 'Bearer ' + token1}, timeout=300) as http:
            created = http.post('/conversations')
            created.raise_for_status()
            cid = created.json()['conversation_id']
            report['conversation_id'] = cid
            save('development_cases.json', {'cases': [{'message_id': mid, 'content': text} for mid, text in cases],
                'review_expectations': selected['expected'] if selected else ['M1 needs_user for unknown deployment', 'M2 host port 5433, no definite root cause',
                    'M3 separate task with container/network facts, no inherited published port', 'M4 resumes first task',
                    'M5 asks task attribution without RAG', 'M6 resolves first task and continues'],
                'expectations_sent_to_model': False})
            for mid, text in cases:
                started = monotonic()
                response = http.post(f'/conversations/{cid}/messages', json={'message_id': mid, 'content': text})
                elapsed = monotonic() - started
                body = response.json()
                history = store.read_history('U1', cid)
                last_run = next((r for r in history['runs'] if r['input_version'] == body.get('input_version')), None)
                preparation = next((p for p in history['preparations'] if p['input_version'] == body.get('input_version')), None)
                record = {'message_id': mid, 'http_status': response.status_code, 'response': body,
                    'task_id': last_run['input_snapshot']['task']['task_id'] if last_run else None,
                    'turn_question': last_run['input_snapshot']['turn']['question'] if last_run else None,
                    'retrieval_question': last_run['input_snapshot']['turn'].get('retrieval_question') if last_run else None,
                    'question_anchor': last_run['input_snapshot']['turn'].get('question_anchor') if last_run else None,
                    'analysis_scope': preparation.get('analysis_scope') if preparation else None,
                    'all_database_connections_closed': all(c.closed for c in connections),
                    'attempt_counts': deepcopy(accounting.data['attempts']), 'elapsed_seconds': round(elapsed, 3),
                    'context_facts': [{'target': f['target'], 'value': f['value']}
                        for f in (last_run['input_snapshot']['context']['current_facts'] if last_run else [])]}
                report['turns'].append(record)
                save(mid + '_response.json', record)
                save(mid + '_history.json', history)
                save('verification.json', report)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if body.get('error_code') == 'provider_balance_insufficient':
                    report['remaining_messages_not_run'] = [m for m, _ in cases[cases.index((mid, text)) + 1:]]
                    report['interruption'] = 'provider_balance_insufficient; no automatic retry'
                    break
                if response.status_code >= 500 and last_run is None:
                    report['remaining_messages_not_run'] = [m for m, _ in cases[cases.index((mid, text)) + 1:]]
                    report['interruption'] = 'preparation_failed; stop dependent turns instead of resubmitting queued messages'
                    break
            before = deepcopy(accounting.data['attempts'])
            last = report['turns'][-1]
            if report.get('interruption') or last['response'].get('status') in ('failed','running','preparing'):
                report['replay'] = {'status': 'not_run', 'reason': 'provider_or_preparation_failure; no automatic retry'}
            else:
                sent = next((mid, text) for mid, text in cases if mid == last['message_id'])
                replay = http.post(f'/conversations/{cid}/messages', json={'message_id': sent[0], 'content': sent[1]})
                report['replay'] = {'http_status': replay.status_code, 'replayed': replay.json().get('replayed'),
                                    'no_new_provider_calls': before == accounting.data['attempts']}
            denied = http.get(f'/conversations/{cid}/messages', headers={'Authorization': 'Bearer ' + token2})
            report['other_owner_history_http_status'] = denied.status_code
            save('public_history.json', http.get(f'/conversations/{cid}/messages').json())
        report.update(status='finished', attempts=deepcopy(accounting.data['attempts']),
            pinned_build_ids=[b.build.build_id for b in runtime['bindings']],
            retrieval_trace=[b.retriever.trace for b in runtime['bindings']],
            source_unchanged=all(hashlib.sha256((project / 'app' / n).read_bytes()).hexdigest() == h for n, h in hashes.items()))
        if selected:
            report['suite_unchanged'] = hashlib.sha256(args.suite.read_bytes()).hexdigest() == report['suite_sha256']
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), flush=True)
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=12)
            report['http_server_stopped'] = not thread.is_alive()
        if (cluster / 'postmaster.pid').exists():
            with (output / 'shutdown.log').open('w', encoding='utf-8') as log:
                stopped = subprocess.run([str(commands['pg_ctl']), '-D', str(cluster), '-m', 'fast', '-w', '-t', '20', 'stop'],
                    stdout=log, stderr=subprocess.STDOUT, timeout=25, creationflags=flags)
                report['database_shutdown_exit_code'] = stopped.returncode
        report['database_stopped'] = not (cluster / 'postmaster.pid').exists()
        save('verification.json', report)
        print(json.dumps({k: report.get(k) for k in ('status','attempts','semantic_review','http_server_stopped','database_stopped')}), flush=True)


if __name__ == '__main__':
    main()
