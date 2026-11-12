"""Incident selection and evidence constraints; no model calls or SQL here."""
from dataclasses import dataclass
from uuid import uuid4

from app.conversation_context import UserMessage
from app.task_constraints import (ConstraintSnapshot, IncidentRef, apply_evidence_updates,
                                  project_incident_constraints)


CONSTRAINT_PREPARATION_PROMPT = '''
联合JSON现在还必须有evidence_request，与updates、analysis_request处于同一顶层。
只有这一种形状（字段都必须存在，无值用null或[]）：
"evidence_request": {
  "incident": {"operation":"create|select|general|clarify", "incident_id":null,
    "description":null,"question":null,"candidate_incident_ids":[],
    "source":{"message_id":"当前消息ID","quote":"当前连续原文"}},
  "resolves_input_version":null,
  "updates":[{"evidence_kind":"incident_logs|incident_error_details|other_incident_evidence",
    "availability":"not_obtained|unavailable|available|uncertain",
    "previous_constraint_id":null,"source":{"message_id":"当前消息ID","quote":"连续原文"}}]
}
任务与故障事件不同：同任务可含多次故障；同次故障的多条补充沿用事件，不按消息新建。
incident_catalog仅是当前选中任务已登记的事件；constraint_state是该任务的现有约束。
create：用户明确在讨论一个尚未登记的实际故障，description给简短事件描述，ID由程序产生。
select：选择目录中所指事件的incident_id，description/question为null，候选列表为空。
general：仅讨论一般知识/配置规则或假设，无需绑定真实事件；ID/description/question均null，updates=[]。
clarify：有多个合理事件归属或归属不清，question具体追问，candidate_incident_ids仅列已登记候选，
其他字段null，updates=[]。不要猜一个事件去更新，不要把历史证据限制改成新故障。
source引用当前用户原话；resolved_original_message仅在服务器提供时可作为额外来源。
pending_incidents是尚未明确归属的旧原话；当前用户明确解决其中一条时，resolves_input_version
填其版本，事件选择source仍用当前原话，updates允许引用该条旧原话的ID及原文，不能冒充当前来源。
没有解决旧归属时resolves_input_version=null，不得引用任意其他历史作为新的约束依据。
updates只描述用户明确的证据可取得性，不写环境事实、原因猜测或诊断结论。
没拿到不等于永久不可得；日志清理不等于所有其他证据不可得，除非原话明确如此。
日志、报错及其他现场证据分别更新；无相关陈述不更新，明确再次确认同值可提交更新，
程序保留最初来源并保存本次提议。旧消息不能覆盖之后明确更新/再次确认的状态。前记录ID从constraint_state复制，
无前记录时null，不自造ID。只找回部分文件不能宣告整类可用，不确定可保留uncertain。
general不删除已有约束；约束只是用户陈述，不是所有后续问题的一律拒答规则。
'''


@dataclass(frozen=True)
class PreparedConstraints:
    scope: dict
    state: ConstraintSnapshot
    registration: dict | None = None


def resolve_evidence_request(request, message, *, task_id, state, catalog=(), pending=(),
                             deferred_message=None):
    if (not isinstance(request, dict) or set(request) != {'incident','resolves_input_version','updates'}
            or not isinstance(request['updates'], list) or not isinstance(request['incident'], dict)):
        raise ValueError('invalid_evidence_request')
    incident = request['incident']
    if set(incident) != {'operation','incident_id','description','question','candidate_incident_ids','source'}:
        raise ValueError('invalid_incident_request')
    if not isinstance(task_id, str) or not task_id.strip() or not isinstance(state, ConstraintSnapshot):
        raise ValueError('require_selected_task_constraint_state')
    if message.input_version <= state.input_version:
        raise ValueError('constraint_version_not_newer')
    allowed_sources = {message.message_id: message}
    if deferred_message is not None:
        allowed_sources[deferred_message.message_id] = deferred_message
    source = incident['source']
    if not isinstance(source, dict) or set(source) != {'message_id','quote'}:
        raise ValueError('invalid_incident_source')
    canonical = allowed_sources.get(source['message_id']) if isinstance(source['message_id'], str) else None
    if (canonical is None or not isinstance(source['quote'], str) or not source['quote'].strip()
            or source['quote'] not in canonical.content):
        raise ValueError('incident_source_not_authorized')
    entries = {r['incident_id']: r for r in catalog if r['task_id'] == task_id}
    candidates = incident['candidate_incident_ids']
    if (not isinstance(candidates, list) or any(not isinstance(i, str) or i not in entries for i in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError('incident_candidate_not_in_selected_task')
    resolved_version = request['resolves_input_version']
    deferred = [deferred_message] if deferred_message is not None else []
    if resolved_version is not None:
        previous = next((p for p in pending if p['task_id'] == task_id
                         and p['input_version'] == resolved_version), None)
        if type(resolved_version) is not int or previous is None:
            raise ValueError('incident_resolution_not_pending')
        original = UserMessage(previous['message_id'], previous['input_version'], previous['content'])
        if original.input_version >= message.input_version:
            raise ValueError('invalid_constraint_deferred_source')
        if not any(m.message_id == original.message_id for m in deferred):
            deferred.append(original)
    mode = incident['operation']
    if mode not in ('create','select','general','clarify'):
        raise ValueError('invalid_incident_operation')
    if mode == 'clarify':
        if (incident['incident_id'] is not None or incident['description'] is not None
                or not isinstance(incident['question'], str) or not incident['question'].strip()
                or request['updates'] or resolved_version is not None):
            raise ValueError('ambiguous_incident_cannot_update_constraints')
        return PreparedConstraints({'mode':'needs_clarification','question':incident['question'],
            'candidate_incident_ids':candidates}, ConstraintSnapshot(message.input_version, state.current))
    if incident['question'] is not None or candidates:
        raise ValueError('unexpected_incident_clarification_fields')
    registration = None
    if mode == 'general':
        if (incident['incident_id'] is not None or incident['description'] is not None
                or request['updates'] or resolved_version is not None):
            raise ValueError('general_scope_has_no_incident_updates')
        return PreparedConstraints({'mode':'general'}, ConstraintSnapshot(message.input_version,state.current))
    if mode == 'create':
        if (incident['incident_id'] is not None or not isinstance(incident['description'],str)
                or not incident['description'].strip()):
            raise ValueError('invalid_new_incident')
        iid = uuid4().hex
        registration = {'task_id':task_id,'incident_id':iid,'description':incident['description'],
                        'source_message_id':source['message_id'],'source_quote':source['quote']}
    else:
        iid = incident['incident_id']
        if not isinstance(iid,str) or iid not in entries or incident['description'] is not None:
            raise ValueError('incident_not_in_selected_task')
    ref = IncidentRef(task_id,iid)
    registry = frozenset({IncidentRef(task_id,key) for key in entries} | {ref})
    updated = apply_evidence_updates(state,message,incident=ref,updates=request['updates'],
        allowed_incidents=registry,deferred_messages=tuple(deferred))
    return PreparedConstraints({'mode':'incident','task_id':task_id,'incident_id':iid},updated,registration)


def project_prepared_constraints(state, scope, task_id):
    if scope == {'mode':'general'}:
        return {'input_version':state.input_version,'scope':scope,'evidence_constraints':[]}
    if (not isinstance(scope,dict) or set(scope) != {'mode','task_id','incident_id'}
            or scope['mode'] != 'incident' or scope['task_id'] != task_id):
        raise ValueError('incident_scope_not_ready')
    ref = IncidentRef(task_id,scope['incident_id'])
    result = project_incident_constraints(state,ref,allowed_incidents=frozenset({ref}))
    return {'input_version':state.input_version,'scope':scope,'evidence_constraints':result['evidence_constraints']}
