"""Message -> preparation -> pinned RAG -> version-guarded publication.

Synchronous provider/SQL calls run from synchronous HTTP handlers. The database
claims an execution in a short transaction; no lock/connection spans model I/O.
"""
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from app.context_proposal import RegisteredTarget
from app.conversation_context import FactTarget
from app.conversation_preparation import prepare_next_turn
from app.conversation_attempts import RunAttempts, AttemptHeartbeat, lease_duration
from app.provider_errors import ModelProviderError, provider_public_reply


@dataclass(frozen=True)
class ConversationExecution:
    build_id: str
    execute: Callable[[dict], dict]

    def __post_init__(self):
        if not isinstance(self.build_id, str) or not self.build_id.strip() or not callable(self.execute):
            raise ValueError('invalid_conversation_execution')


def deployment_registry(conversation_id, message_id):
    """A server-assigned namespace for one application/database incident.

This profile has a fixed environment vocabulary, not automatic discovery of an
arbitrary multi-service topology. Returning to a task uses its stored registry.
"""
    scope = 'task_' + sha256((conversation_id + '\0' + message_id).encode()).hexdigest()[:24]
    fields = (
        ('application', 'runtime_location', '该任务中应用的运行位置；宿主机为host，容器为container'),
        ('database', 'runtime_location', '该任务中数据库的运行位置；明确Compose为compose，仅容器为container，宿主机为host'),
        ('database', 'published_host_port', '数据库映射到宿主机的端口；只保存用户明确提供的字符串'),
        ('database', 'container_port', '数据库容器端口；从用户明确提供的HOST_PORT:CONTAINER_PORT映射读取右侧端口的数字字符串，不使用左侧宿主机端口，不从默认端口或文档示例猜测'),
        ('database', 'service_name', 'Compose中的数据库服务名；只保存用户明确提供的名称'),
        ('application', 'network_relation', '用户明确说应用与数据库在同一个Compose网络时保存same_compose_network；明确分属不同网络时保存different_network；未陈述网络关系则不添加，不因都在容器或同一宿主机就认定同网'),
        ('database', 'host_relation', '数据库所在宿主机与应用宿主机的关系，如明确在同一台机器；不猜测'),
        ('database', 'host_address', '数据库所在宿主机的明确地址；不从例子或常识填入'),
        ('application', 'error_text', '实际报错的具体连续原文；证据未拿到/不可取得属于evidence_request，不能当报错'),
        ('application', 'configuration_text', '用户明确提供的当前配置原文；不能填入助手建议的配置'),
    )
    contracts = {'published_host_port': 'port', 'container_port': 'port', 'service_name': 'service_name',
                 'network_relation': 'network_relation', 'host_address': 'source_literal',
                 'error_text': 'error_literal', 'configuration_text': 'source_literal'}
    return tuple(RegisteredTarget(FactTarget(entity, scope, field), description,
                 ('app_location' if entity == 'application' else 'db_location') if field == 'runtime_location'
                 else contracts.get(field))
                 for entity, field, description in fields)


class ConversationService:
    def __init__(self, store, *, generate_task, generate_context, bind_execution,
                 new_registry=deployment_registry, resume_execution=None, lease_seconds=90,
                 include_constraints=False):
        if not all(callable(fn) for fn in (generate_task, generate_context, bind_execution, new_registry)):
            raise ValueError('conversation_dependencies_must_be_callable')
        self.store = store
        self.generate_task, self.generate_context = generate_task, generate_context
        self.bind_execution, self.new_registry = bind_execution, new_registry
        if resume_execution is not None and not callable(resume_execution):
            raise ValueError('resume_execution_must_be_callable')
        self.resume_execution, self.lease_seconds = resume_execution, lease_duration(lease_seconds)
        self.attempts = RunAttempts(store)
        if type(include_constraints) is not bool:
            raise ValueError('invalid_constraint_profile')
        self.include_constraints = include_constraints

    def _execute_run(self, owner_id, conversation_id, run_id, *, execution=None):
        attempt = self.attempts.claim(owner_id, conversation_id, run_id, lease_seconds=self.lease_seconds)
        if not attempt['acquired']:
            return
        with AttemptHeartbeat(self.attempts, owner_id, conversation_id, run_id, attempt['attempt_id'], self.lease_seconds):
            stage = 'bind'
            try:
                stored = self.store.read_run_input(owner_id, conversation_id, run_id)
                if execution is None:
                    if self.resume_execution is None:
                        raise ValueError('resume_execution_not_configured')
                    execution = self.resume_execution(stored['build_id'])
                if not isinstance(execution, ConversationExecution) or execution.build_id != stored['build_id']:
                    raise ValueError('recovery_build_mismatch')
                stage = 'execute'
                result = execution.execute(stored['input_snapshot'])
                if not isinstance(result, dict) or result.get('status') not in (
                        'answer', 'needs_user', 'insufficient', 'stopped', 'failed'):
                    raise ValueError('invalid_conversation_execution_result')
            except ModelProviderError as error:
                result = {'status': 'failed', 'message': '模型服务暂时不可用。', 'required_information': [],
                          'error': error.detail(stage)}
            except Exception as error:
                result = {'status': 'failed', 'message': '本轮回答暂时未能完成。', 'required_information': [],
                    'error': {'stage': stage, 'code': 'execution_timeout' if isinstance(error, TimeoutError)
                              else 'execution_binding_failed' if stage == 'bind' else 'execution_failed',
                              'error_type': type(error).__name__}}
            self.attempts.finish(owner_id, conversation_id, run_id, attempt['attempt_id'], result,
                                  status='failed' if result['status'] == 'failed' else 'completed')

    def _registry(self, owner_id, conversation_id, decision):
        if decision.selection['operation'] == 'create':
            return self.new_registry(conversation_id, decision.turn_request['source']['message_id'])
        catalog = self.store.read_preparation(owner_id, conversation_id)['task_catalog']
        task = next((t for t in catalog if t['task_id'] == decision.selection['task_id']), None)
        if task is None or task['registered_targets'] is None:
            # Earlier prototypes require an explicit server migration; do not
            # guess which old fields belong to this task.
            raise ValueError('task_registry_not_configured')
        return tuple(RegisteredTarget(FactTarget(**r['target']), r['description'], r.get('value_contract'))
                     for r in task['registered_targets'])

    def submit(self, owner_id, conversation_id, message_id, content):
        receipt = self.store.append_message(owner_id, conversation_id, message_id, content)
        target = receipt['input_version']
        record = self.store.read_message_turn(owner_id, conversation_id, message_id)
        if record['run'] is not None:
            if record['run']['status'] == 'running' and target == record['current_version']:
                self._execute_run(owner_id, conversation_id, record['run']['run_id'])
                record = self.store.read_message_turn(owner_id, conversation_id, message_id)
            return record, receipt['replayed']
        if record['selection'] is not None and record['selection']['operation'] == 'clarify':
            return record, receipt['replayed']
        if target != record['current_version']:
            return record, receipt['replayed']
        while True:
            view = self.store.read_preparation(owner_id, conversation_id)
            if view['state'].input_version >= target:
                break
            dialogue = self.store.read_dialogue(owner_id, conversation_id, view['state'].input_version)
            try:
                prepare_next_turn(self.store, owner_id, conversation_id,
                    generate_task=self.generate_task, generate_context=self.generate_context,
                    resolve_registry=lambda decision: self._registry(owner_id, conversation_id, decision),
                    recent_dialogue=dialogue, include_constraints=self.include_constraints)
            except ValueError:
                # Another request may have committed the same pending message
                # while this caller was waiting for its model response.
                if self.store.read_preparation(owner_id, conversation_id)['state'].input_version > view['state'].input_version:
                    continue
                raise
        record = self.store.read_message_turn(owner_id, conversation_id, message_id)
        if (record['run'] is not None or record['current_version'] != target
                or record['selection']['operation'] == 'clarify'
                or (record.get('analysis_scope') or {}).get('mode') == 'needs_clarification'
                or (record.get('incident_scope') or {}).get('mode') == 'needs_clarification'):
            return record, receipt['replayed']
        execution = self.bind_execution()
        if not isinstance(execution, ConversationExecution):
            raise ValueError('binder_must_return_conversation_execution')
        try:
            claimed = self.store.claim_prepared_run(owner_id, conversation_id, target, execution.build_id)
        except ValueError as error:
            if str(error) == 'conversation_changed':
                return self.store.read_message_turn(owner_id, conversation_id, message_id), receipt['replayed']
            # A concurrent caller can already have pinned a different active
            # build. Reuse its receipt instead of starting a second execution.
            existing = self.store.read_message_turn(owner_id, conversation_id, message_id)
            if str(error) == 'run_input_conflict' and existing['run'] is not None:
                return existing, receipt['replayed']
            raise
        self._execute_run(owner_id, conversation_id, claimed['run_id'], execution=execution)
        return self.store.read_message_turn(owner_id, conversation_id, message_id), receipt['replayed']

    def history(self, owner_id, conversation_id):
        history = self.store.read_history(owner_id, conversation_id)
        latest = max((m['input_version'] for m in history['messages']), default=0)
        prepared = {p['input_version']: p for p in history['preparations']}
        runs = {r['input_version']: r for r in history['runs']}
        return [dict(conversation_id=conversation_id, message_id=m['message_id'], input_version=m['input_version'],
            current_version=latest, content=m['content'], selection=prepared.get(m['input_version'], {}).get('selection'),
            analysis_scope=prepared.get(m['input_version'], {}).get('analysis_scope'),
            incident_scope=prepared.get(m['input_version'], {}).get('incident_scope'),
            turn_question=prepared.get(m['input_version'], {}).get('turn_question'), run=runs.get(m['input_version']))
            for m in history['messages']]


def public_turn(record, *, replayed=False, historical=False):
    """Public projection: source citations and messages, no raw model/SQL trace."""
    current = record['input_version'] == record['current_version']
    body = {'conversation_id': record['conversation_id'], 'message_id': record['message_id'],
        'input_version': record['input_version'], 'status': 'pending', 'message': '消息已保存，正在处理。',
        'required_information': [], 'sources': [], 'replayed': replayed, 'is_current': False}
    if not current and not historical:
        body.update(status='superseded', message='已有更新的补充信息，请查看较新消息的处理结果。')
        return 200, body
    selection, run = record['selection'], record['run']
    if selection and selection['operation'] == 'clarify':
        body.update(status='needs_user', message=selection['question'],
                    required_information=[selection['question']], is_current=current)
        return 200, body
    scope = record.get('analysis_scope') or {}
    if scope.get('mode') == 'needs_clarification':
        body.update(status='needs_user', message=scope['question'],
                    required_information=[scope['question']], is_current=current)
        return 200, body
    incident = record.get('incident_scope') or {}
    if incident.get('mode') == 'needs_clarification':
        body.update(status='needs_user', message=incident['question'],
                    required_information=[incident['question']], is_current=current)
        return 200, body
    if run is None or run['status'] == 'running':
        return 202, body
    result = run['result']
    if run['status'] == 'failed':
        provider_reply = provider_public_reply(result.get('error') or {})
        if provider_reply is not None:
            body.update(provider_reply)
            return 503, body
        timeout = (result.get('error') or {}).get('code', '').endswith('timeout')
        body.update(status='failed', message='本轮回答暂时未能完成；已确认的上下文已保留。')
        return 503 if timeout else 500, body
    body.update(status=result['status'], message=result['message'],
        required_information=result.get('required_information', []),
        sources=(result.get('answer') or {}).get('sources', []), is_current=current)
    return 200, body
