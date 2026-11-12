"""Freeze and run reviewed multi-turn development cases using the existing probe.

Each case has its own disposable PostgreSQL/HTTP service. The existing probe
stops both. Completion/action checks are not semantic accuracy judgments.
No keys or complete environment variables are written to the manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import mkdtemp


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_same_fingerprint(frozen, current):
    if frozen != current:
        raise ValueError('suite_runtime_or_corpus_changed')


def classify_report(case, report):
    """Separate operational observations and action checks from manual review."""
    turns = report.get('turns', [])
    observed = {t['message_id']: t for t in turns}
    checks = [{'message_id': item['message_id'], 'expected': item['action'],
               'actual': observed.get(item['message_id'], {}).get('response', {}).get('status'),
               'matches': observed.get(item['message_id'], {}).get('response', {}).get('status') == item['action']}
              for item in case['expected']]
    failures = [{'message_id': t['message_id'], 'http_status': t['http_status'],
                 'error_code': t['response'].get('error_code'), 'status': t['response'].get('status')}
                for t in turns if t['http_status'] >= 400 or t['response'].get('status') == 'failed']
    missing = [t['message_id'] for t in case['turns'] if t['message_id'] not in observed]
    if report.get('source_unchanged') is not True or report.get('suite_unchanged') is not True:
        status = 'incomparable'
    elif missing or report.get('interruption'):
        status = 'interrupted'
    elif failures:
        status = 'request_failures'
    elif not all(c['matches'] for c in checks):
        status = 'action_mismatch'
    else:
        status = 'responses_observed_pending_semantic_review'
    return {'status': status, 'planned_turns': len(case['turns']), 'observed_turns': len(turns),
            'missing_messages': missing, 'request_failures': failures, 'action_checks': checks,
            'semantic_review': 'pending', 'attempts': report.get('attempts'),
            'resources_stopped': report.get('http_server_stopped') is True and report.get('database_stopped') is True}


def public_configuration():
    from urllib.parse import urlparse
    from app.ask import create_generation_client, generation_options
    from app.embedding_runtime import BASE_URL, MODEL, DIMENSIONS
    def endpoint(url):
        value = urlparse(str(url))
        return {'scheme': value.scheme, 'host': value.hostname, 'port': value.port, 'path': value.path}
    client, model = create_generation_client()
    with client:
        return {'generation_model': model, 'generation_endpoint': endpoint(client.base_url),
                'generation_options': generation_options(client),
                'evaluation_options': generation_options(client, evaluate=True),
                'embedding_model': MODEL, 'embedding_dimension': DIMENSIONS,
                'embedding_endpoint': endpoint(BASE_URL)}


def fingerprint(project, suite_path, release_path=None):
    from app.build_runtime import BuildManifest, load_build
    release_path = release_path or project / os.getenv('RAG_CONVERSATION_RELEASE_PATH',
        'data/corpus_releases/deploy_faults_20261102_v1/release.json')
    release = json.loads(release_path.read_text(encoding='utf-8'))
    build = BuildManifest(**release['candidate'])
    load_build(build, release_path.parent / release['artifact_root'])
    return {'sources': {str(p.relative_to(project)): sha(p) for p in sorted((project / 'app').rglob('*.py'))},
            'suite_sha256': sha(suite_path), 'release_sha256': sha(release_path),
            'knowledge_build': release['candidate'], 'providers': public_configuration()}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--suite', type=Path, default=Path('data/evaluations/multiturn_development_v1.json'))
    parser.add_argument('--pg-bin', type=Path, required=True)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv
    load_dotenv(project / '.env', override=False)
    release_path = (args.release or project / os.getenv('RAG_CONVERSATION_RELEASE_PATH',
        'data/corpus_releases/deploy_faults_20261102_v1/release.json')).resolve()
    from app.run_conversation_http_probe import load_reviewed_case
    suite = json.loads(args.suite.read_text(encoding='utf-8'))
    cases = suite['cases']
    for case in cases:
        load_reviewed_case(args.suite, case['case_id'])
    root = project / 'data/architecture_demos'
    root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='multiturn_suite_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_'), dir=root))
    frozen_suite = output / 'frozen_suite.json'
    frozen_suite.write_bytes(args.suite.read_bytes())
    frozen = fingerprint(project, frozen_suite, release_path)
    if frozen['knowledge_build']['build_id'] != suite['knowledge_build_id']:
        raise ValueError('suite_knowledge_build_mismatch')
    for name in frozen['sources']:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((project / name).read_bytes())
    versions = {}
    for name in ('openai', 'httpx', 'fastapi', 'pydantic', 'psycopg', 'uvicorn', 'numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output / 'manifest.json', {'fingerprint': frozen, 'python': sys.version,
        'platform': platform.platform(), 'package_versions': versions,
        'case_ids': [c['case_id'] for c in cases], 'planned_turns': sum(len(c['turns']) for c in cases),
        'credentials_saved': False, 'expected_labels_sent_to_models': False,
        'hosted_model_weights_immutable': 'not guaranteed by an alias/configuration snapshot'})
    report = {'status': 'running', 'cases': [], 'semantic_review': 'pending',
              'overall_accuracy_claim': False, 'case_retries': 0, 'manifest': str(output / 'manifest.json')}
    print('SUITE_OUTPUT ' + str(output), flush=True)
    try:
        for case in cases:
            require_same_fingerprint(frozen, fingerprint(project, frozen_suite, release_path))
            cid = case['case_id']
            log_path = output / (cid + '.log')
            with log_path.open('w', encoding='utf-8') as log:
                child = subprocess.run([sys.executable, '-m', 'app.run_conversation_http_probe', '--live',
                    '--suite', str(frozen_suite), '--case-id', cid, '--label', 'candidate', '--pg-bin', str(args.pg_bin),
                    '--release', str(release_path)],
                    cwd=project, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            found = [line.removeprefix('实验记录目录: ').strip() for line in lines if line.startswith('实验记录目录: ')]
            if len(found) != 1:
                item = {'case_id': cid, 'status': 'probe_launch_failed', 'exit_code': child.returncode,
                        'log': str(log_path), 'semantic_review': 'not_judged'}
            else:
                directory = Path(found[0]).resolve()
                if not directory.is_relative_to(root.resolve()):
                    raise ValueError('case_output_outside_experiment_root')
                case_report = json.loads((directory / 'verification.json').read_text(encoding='utf-8'))
                item = {'case_id': cid, 'directory': str(directory), 'exit_code': child.returncode,
                        **classify_report(case, case_report)}
                if not item['resources_stopped']:
                    report['cases'].append(item)
                    raise ValueError('case_services_not_stopped')
            require_same_fingerprint(frozen, fingerprint(project, frozen_suite, release_path))
            report['cases'].append(item)
            write_json(output / 'verification.json', report)
            print(json.dumps({'case_id': cid, 'status': item['status'], 'completed_cases': len(report['cases']),
                              'planned_cases': len(cases)}, ensure_ascii=False), flush=True)
            if any(e.get('error_code') == 'provider_balance_insufficient' for e in item.get('request_failures', [])):
                report['status'] = 'interrupted_provider_balance_insufficient'
                report['remaining_case_ids'] = [c['case_id'] for c in cases[len(report['cases']):]]
                break
        if report['status'] == 'running':
            report['status'] = 'execution_complete_pending_semantic_review'
        report['fingerprint_unchanged'] = True
    except BaseException as error:
        report.update(status='interrupted', interruption=type(error).__name__ + ':' + str(error))
        raise
    finally:
        write_json(output / 'verification.json', report)
        print(json.dumps({'status': report['status'], 'observed_cases': len(report['cases']),
                          'semantic_review': report['semantic_review']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
