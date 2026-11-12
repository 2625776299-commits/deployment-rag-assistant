"""Coordinate task recognition and scoped context proposals before one commit.

The authenticated caller provides owner_id and a trusted target-registry
resolver. Neither model may define permissions or invent registered targets.
No model is invoked inside a database transaction. RAG/HTTP follow this stage.
"""
import json

from app.context_proposal import (PreparedContext, RegisteredTarget, build_context_update_messages,
                                 validate_context_value_contracts)
from app.conversation_context import ContextState, FactTarget, apply_context_updates, parse_context_updates
from app.task_proposal import PreparedTaskTurn, prepare_task_turn
from app.analysis_preparation import (
    ANALYSIS_PREPARATION_PROMPT, resolve_analysis_request, guard_hypothetical_context_sources,
)
from app.constraint_preparation import CONSTRAINT_PREPARATION_PROMPT, resolve_evidence_request


def prepare_task_context(state, message, task: PreparedTaskTurn, *, registry, generate,
                         recent_dialogue=(), include_analysis=False, analysis_history=(),
                         constraint_view=None) -> PreparedContext:
    if task.selection['operation'] == 'clarify':
        raise ValueError('ambiguous_task_cannot_propose_context')
    if constraint_view is not None and not include_analysis:
        raise ValueError('constraints_require_joint_preparation')
    if (not isinstance(registry, tuple)
            or any(not isinstance(item, RegisteredTarget) or not isinstance(item.target, FactTarget)
                   for item in registry)):
        raise ValueError('invalid_context_registry')
    if not registry and not include_analysis and constraint_view is None:
        proposal = {'updates': []}
        return PreparedContext(proposal, apply_context_updates(state, message, proposal, allowed_targets=frozenset()))
    # Keep unrelated state in storage, but do not present it as the selected
    # task's environment. The existing validator checks registry types/IDs.
    targets = frozenset(item.target for item in registry)
    projected = ContextState(state.input_version,
        tuple(f for f in state.current_facts if f.target in targets), (),
        tuple(p for p in state.pending_facts if p.target in targets))
    messages = build_context_update_messages(projected, message, registry=registry,
                                               recent_dialogue=recent_dialogue)
    payload = json.loads(messages[1]['content'])
    payload['selected_task'] = {'task_id': task.selection.get('task_id'),
                               'goal': task.task_goal, 'turn_question': task.original_question or task.question,
                               'question_interpretation': task.question}
    if include_analysis:
        payload['analysis_history'] = list(analysis_history)
        messages[0]['content'] = messages[0]['content'].replace(
            '只输出一个JSON对象：{"updates": [...]}。不输出分析或诊断。',
            '只输出环境更新与analysis_request的联合JSON对象，不输出分析或诊断。').replace(
            '没有可应用的变化时返回{"updates": []}，不伪造变化。',
            '没有实际环境变化时updates用[]，仍必须输出analysis_request，不伪造变化。')
        analysis_prompt = ANALYSIS_PREPARATION_PROMPT
        if constraint_view is not None:
            analysis_prompt = analysis_prompt.replace(
                '{"updates":[...],"analysis_request":{...}}',
                '{"updates":[...],"analysis_request":{...},"evidence_request":{...}}')
            example = '{"updates":[],"analysis_request":{"mode":"current","source":{"message_id":"本轮ID","quote":"本轮原话"},"confirmation_from_input_version":null}}'
            analysis_prompt = analysis_prompt.replace(
                '完整结构示例（字段中的占位原文必须替换为本轮真实ID和原文）：\n' + example,
                '联合顶层三个字段均必须存在；analysis_request本身的字段严格按mode选择。')
        messages[0]['content'] += analysis_prompt
    if constraint_view is not None:
        task_id = task.selection.get('task_id')
        payload['incident_catalog'] = [r for r in constraint_view['incident_catalog'] if r['task_id'] == task_id]
        payload['constraint_state'] = [c for c in constraint_view['constraint_state'].to_dict()['current']
                                       if c['incident']['task_id'] == task_id]
        payload['pending_incidents'] = [r for r in constraint_view['pending_incidents'] if r['task_id'] == task_id]
        messages[0]['content'] += CONSTRAINT_PREPARATION_PROMPT
    messages[0]['content'] += '''
selected_task只说明本次选中的任务和本轮问题，不是新的现场事实来源。
question_interpretation是模型解释，不得据此把原话的疑问改成已成立的条件；以当前原文为依据。
仅更新registered_targets登记的对象/环境；其他任务的信息由服务器保存，不要挪入本任务。
不要从任务目标或本轮问题的改写中提取现场事实，事实依据必须回到用户原始消息。
'''
    if task.deferred_message is not None:
        deferred = task.deferred_message
        payload['resolved_original_message'] = {
            'message_id': deferred.message_id, 'input_version': deferred.input_version,
            'content': deferred.content,
        }
        messages[0]['content'] += '''
本轮已明确解决一条旧消息的任务归属：resolved_original_message。
来源规则在此扩展：除当前消息外，仅允许额外引用这一条原消息的ID与连续原文。
它保存原始内容，本轮回复保存归属依据，不能伪造为本轮原文。不得引用其他历史。
当前回复明确纠正时使用当前原文；旧消息不能覆盖在它之后更新的环境陈述。
'''
    messages[1]['content'] = json.dumps(payload, ensure_ascii=False)
    def validate(raw):
        analysis_request = evidence_request = None
        if include_analysis:
            if not isinstance(raw, str) or len(raw) > 24000:
                raise ValueError('invalid_analysis_preparation_response')
            try:
                combined = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError('analysis_preparation_is_not_json') from None
            fields = {'updates','analysis_request'} | ({'evidence_request'} if constraint_view is not None else set())
            if not isinstance(combined, dict) or set(combined) != fields:
                raise ValueError('analysis_request_missing_or_extra_fields')
            raw = json.dumps({'updates': combined['updates']}, ensure_ascii=False)
            analysis_request = combined['analysis_request']
            scope = resolve_analysis_request(analysis_request, message, allowed_targets=targets,
                history=analysis_history, deferred_message=task.deferred_message)
            if constraint_view is not None:
                evidence_request = combined['evidence_request']
                resolve_evidence_request(evidence_request,message,
                    task_id=task.selection.get('task_id') or '__new_selected_task__',
                    state=constraint_view['constraint_state'],catalog=payload['incident_catalog'],
                    pending=payload['pending_incidents'],deferred_message=task.deferred_message)
        proposal = parse_context_updates(raw)
        if include_analysis:
            guard_hypothetical_context_sources(proposal, scope)
        updated = apply_context_updates(state, message, proposal, allowed_targets=targets,
                                        deferred_message=task.deferred_message)
        validate_context_value_contracts(proposal, registry, analysis_request=analysis_request,
                                         evidence_request=evidence_request)
        return proposal,updated,analysis_request,evidence_request

    checks = []
    repairable = {'analysis_preparation_is_not_json','analysis_request_missing_or_extra_fields',
        'analysis_source_not_authorized','constraint_quote_not_in_message','incident_source_not_authorized',
        'invalid_evidence_request','invalid_incident_request','repeated_hypothesis_requires_inheritance',
        'context_value_not_in_registered_enum','context_value_not_a_port','context_value_not_a_service_name',
        'context_literal_not_in_source_quote','evidence_availability_is_not_error_text',
        'previous_fact_does_not_match_target'}
    for attempt in range(2 if constraint_view is not None else 1):
        # Provider errors/timeouts are outside the validation catch: never retry
        # those. Only an obtained bounded malformed proposal gets one correction.
        raw = generate(messages)
        check = {'raw_text':raw if isinstance(raw,str) else None,'validation_error':None}
        checks.append(check)
        try:
            proposal,updated,analysis_request,evidence_request = validate(raw)
        except ValueError as error:
            check['validation_error'] = str(error)
            if attempt or str(error) not in repairable or not isinstance(raw,str) or len(raw)>24000:
                raise
            messages = [*messages,{'role':'assistant','content':raw},
                {'role':'user','content':('准备提议未通过校验：'+str(error)+
                    '。请在完全相同的任务、当前原话、已授权历史与登记字段下重新输出完整联合JSON。'
                    '修正来源、字段含义或输出结构；source.quote须逐字复制相应原话的连续短句。'
                    '报错只存实际错误，证据可取得性放evidence_request；未给具体服务名不填关系词。'
                    '相同整套假设必须inherit授权历史版本，不重建来源；来源不明时clarify。'
                    '目标在current_user_stated_facts中不存在时新增用add与previous_fact_id=null；'
                    'replace必须逐字使用该目标当前的fact_id，不能引用其他任务/其他字段。'
                    '必须包含updates、analysis_request、evidence_request；不自造任务/事件/来源ID，'
                    '不删除真实约束、不虚构变更或降低校验要求。') }]
        else:
            return PreparedContext(proposal,updated,analysis_request,evidence_request,tuple(checks))


def prepare_next_turn(store, owner_id, conversation_id, *, generate_task, generate_context,
                      resolve_registry, recent_dialogue=(), include_constraints=False) -> dict:
    """Prepare one pending message; callers reread committed state for execution.

resolve_registry is server-owned, not a model/tool-supplied ID mapping. Its
registry is persisted with the atomic preparation; new entities are not inferred here.
Concurrent identical input may be proposed twice; repository checks prevent
conflicting commits. Model failures leave the original message unprepared.
"""
    view = store.read_preparation(owner_id, conversation_id)
    if view['next_message'] is None:
        return {'status': 'nothing_to_prepare', 'preparation': view}
    decision = prepare_task_turn(view, generate=generate_task, recent_dialogue=recent_dialogue)
    message = view['next_message']
    if decision.selection['operation'] == 'clarify':
        proposal, targets = {'updates': []}, frozenset()
        registry = None
        analysis_request = None
        evidence_request = None
    else:
        registry = resolve_registry(decision)
        context = prepare_task_context(view['state'], message, decision, registry=registry,
            generate=generate_context, recent_dialogue=recent_dialogue, include_analysis=True,
            analysis_history=tuple(r for r in view['analysis_catalog']
                                  if r['task_id'] == decision.selection.get('task_id')),
            constraint_view=view if include_constraints else None)
        proposal = context.proposal
        analysis_request = context.analysis_request
        evidence_request = context.evidence_request
        targets = frozenset(item.target for item in registry)
    return store.commit_preparation(owner_id, conversation_id, message.message_id,
        view['state'].input_version, proposal, decision.selection,
        allowed_targets=targets, turn_request=decision.turn_request, registry=registry,
        analysis_request=analysis_request,evidence_request=evidence_request)
