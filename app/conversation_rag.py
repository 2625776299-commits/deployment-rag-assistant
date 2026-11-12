"""Explicit in-process RAG entry using one already-prepared context snapshot.

The caller selects the current user task and prepares context. No automatic goal
selection, HTTP/session authentication, persistence or provider creation here.
"""
import json
from copy import deepcopy

from app.answering import build_messages, validate_answer
from app.conversation_context import ContextState, FactTarget, UserMessage, restore_context
from app.analysis_scope import (
    AnalysisScope, AnalysisView, ANALYSIS_SCOPE_PROMPT, build_analysis_view,
    project_reasoning_context, restore_analysis_scope,
)
from app.rag_service import run_rag_service
from app.requirement_evaluation import evaluate_requirements
from app.claim_constraints import CLAIM_SCOPE_RULES
from app.task_constraints import restore_constraint_snapshot, IncidentRef
from app.question_authority import restore_question_anchor, QUESTION_AUTHORITY_PROMPT


def _with_question_authority(messages, anchor, interpretation, turn_message):
    if anchor is None:
        return messages
    messages=deepcopy(messages)
    payload=json.loads(messages[1]['content'])
    payload['question_source']={key:value for key,value in anchor.to_dict().items() if key!='text'}
    if interpretation is not None and interpretation!=anchor.text:
        payload['question_interpretation']=interpretation
    if turn_message is not None and turn_message['message_id']!=anchor.message_id:
        payload['current_user_message']=deepcopy(turn_message)
    messages[1]['content']=json.dumps(payload,ensure_ascii=False)
    messages[0]['content']+=QUESTION_AUTHORITY_PROMPT
    return messages


def _validate_task_constraints(payload, input_version, task_id=None):
    if (not isinstance(payload,dict) or set(payload) != {'input_version','scope','evidence_constraints'}
            or payload['input_version'] != input_version or type(payload['input_version']) is not int):
        raise ValueError('invalid_frozen_task_constraints')
    state = restore_constraint_snapshot({'input_version':payload['input_version'],
                                         'current':payload['evidence_constraints']})
    scope = payload['scope']
    if scope == {'mode':'general'}:
        if state.current:
            raise ValueError('general_scope_has_no_incident_updates')
    else:
        if (not isinstance(scope,dict) or set(scope) != {'mode','task_id','incident_id'}
                or scope['mode'] != 'incident' or (task_id is not None and scope['task_id'] != task_id)):
            raise ValueError('frozen_incident_task_mismatch')
        incident = IncidentRef(scope['task_id'],scope['incident_id'])
        if any(c.incident != incident for c in state.current):
            raise ValueError('frozen_incident_constraint_mismatch')
    return deepcopy(payload)


def _with_task_constraints(messages, payload):
    if payload is None:
        return messages
    messages = deepcopy(messages)
    data = json.loads(messages[1]['content'])
    data['task_constraints'] = payload
    messages[1]['content'] = json.dumps(data,ensure_ascii=False)
    messages[0]['content'] += '''
task_constraints是本轮冻结的、只属于所选故障事件的用户证据可获得性陈述，
不是诊断结论或已核验事实。即使question沿用了旧提问，也要使用这些更新后的限制：
not_obtained是尚未取得，unavailable是明确不可取得，available是用户称已取得，
uncertain表示仍不确定；来源和版本不可改写，不把不同事件的限制混用。
不可得的现场证据不能通过文档补检替代，不重复索要该类已不可得资料。
只有请求的具体结论确实依赖这些证据且无其他独立依据时才停止归因；
通用机制或有充分其他证据的请求仍可回答，不能因某条日志不可得一律拒答。
不扩展成用户未请求的新实验或新诊断任务。'''
    return messages


def _validate_context(context):
    project_reasoning_context(context)


def _with_task_background(messages, task_goal):
    if task_goal is None:
        return messages
    annotated = deepcopy(messages)
    payload = json.loads(annotated[1]['content'])
    payload['task_background_goal'] = task_goal
    annotated[1]['content'] = json.dumps(payload, ensure_ascii=False)
    annotated[0]['content'] += '''
question是本轮真正需要回答的问题，包含用户对本轮范围的限制。
task_background_goal只是所属任务的长期背景，不是额外的回答要求或已核验事实。
拆分需求、选择证据、判断覆盖情况和生成回答均以question为准；不能因为没有回答
背景目标的其他部分就判定本轮不完整，也不能恢复其中已被当前上下文纠正的前提。
'''
    return annotated


def build_initial_query(question: str, context: ContextState | AnalysisView) -> str:
    """Deterministic baseline, not a measured query-rewriting optimization.

    Pending/historical values are not appended from the context. The caller
    supplies the current task; extracting it from old mixed-purpose text is not
    implemented here. This baseline is not a context-aware model query planner.
    """
    _validate_context(context)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('invalid_question')
    lines = [question]
    facts = context.baseline_facts if isinstance(context, AnalysisView) else context.current_facts
    for fact in facts:
        lines.append(f'{fact.target.entity_id}/{fact.target.scope_id}/{fact.target.field}: {fact.value}')
    if isinstance(context, AnalysisView) and context.scope.mode == 'hypothetical':
        lines.append('[本轮假设条件，不代表实际部署]')
        for assumption in context.scope.assumptions:
            target = assumption.target
            lines.append(f'[假设] {target.entity_id}/{target.scope_id}/{target.field}: {assumption.value}')
    return '\n'.join(lines)


def build_conversation_answer_messages(question, evidence, context):
    _validate_context(context)
    messages = build_messages(question, evidence)
    payload = json.loads(messages[1]['content'])
    payload['conversation_context'] = project_reasoning_context(context)
    messages[1] = {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}
    messages[0]['content'] += '''
conversation_context提供当前用户陈述与待澄清项，不是已独立核验的现场事实。
对于其中登记的环境字段，以当前陈述或待澄清状态为准；不能恢复原问题中已被
后续纠正或质疑的旧环境值。question仍是本轮要解决的任务，不改成最新补充内容。
只在未知项确实影响本轮结论时追问。通用机制可依资料解释，不因存在无关未知项
一律追问；具体连接地址和实际故障根因不能凭文档示例或合理猜测填入。
回答所需的资料仍必须充分，环境陈述不能替代知识证据。不要重复索要当前已提供的信息。
按本轮问题的范围回答。用户只问host和port时，优先给出与数据库产品无关的
host=...、port=...配置说明。用户未明确数据库产品或驱动时，不能从文档示例、
服务名或默认端口推断产品，也不要添加JDBC、postgres://等特定产品连接串。
材料中的示例使用PostgreSQL不代表用户也使用PostgreSQL；用户给出的5432等端口
只能作为端口事实，不能作为数据库产品已确认的依据。
'''
    messages[0]['content'] += CLAIM_SCOPE_RULES
    if isinstance(context, AnalysisView):
        messages[0]['content'] += ANALYSIS_SCOPE_PROMPT
    return messages


def run_contextual_rag(question: str, context: ContextState, *, retrieve,
                       evaluate_raw, generate_raw, task_goal: str | None = None,
                       analysis_scope: AnalysisScope | None = None,
                       allowed_targets: frozenset[FactTarget] | None = None,
                       task_constraints: dict | None = None, question_anchor: dict | None = None,
                       question_interpretation: str | None = None,
                       turn_message: dict | None = None, **service_options) -> dict:
    """Keep task, initial retrieval query and current context distinct.

    Evaluation proposes relevance/dependencies; code checks consistency, not truth.
    External callbacks receive freshly projected prompts from the same frozen state.
    """
    _validate_context(context)
    anchor=None
    if question_anchor is not None:
        anchor=restore_question_anchor(question_anchor,max_input_version=context.input_version)
        if question!=anchor.text:
            raise ValueError('question_does_not_match_original_source')
        if (not isinstance(turn_message,dict) or set(turn_message)!={'message_id','input_version','content'}):
            raise ValueError('require_canonical_turn_message')
        current=UserMessage(**turn_message)
        if current.input_version!=context.input_version:
            raise ValueError('turn_message_version_mismatch')
        if current.message_id==anchor.message_id and (current.content!=anchor.text
                or current.input_version!=anchor.message_input_version):
            raise ValueError('question_source_conflicts_with_current_message')
    if question_interpretation is not None and (not isinstance(question_interpretation,str)
                                               or not question_interpretation.strip()):
        raise ValueError('invalid_question_interpretation')
    if task_constraints is not None:
        task_constraints = _validate_task_constraints(task_constraints,context.input_version)
    if analysis_scope is not None:
        if isinstance(context, AnalysisView):
            raise ValueError('analysis_view_cannot_be_overlaid_twice')
        context = build_analysis_view(context, analysis_scope, allowed_targets=allowed_targets)
    if task_goal is not None and (not isinstance(task_goal, str) or not task_goal.strip()):
        raise ValueError('invalid_task_background')
    if not all(callable(fn) for fn in (retrieve, evaluate_raw, generate_raw)):
        raise ValueError('contextual_dependencies_must_be_callable')
    reserved = {'evaluate', 'generate', 'initial_query', 'plan_queries'} & service_options.keys()
    if reserved:
        raise ValueError('contextual_entry_owns_stage_bindings')
    final_k = service_options.get('final_k', 3)
    max_chars = service_options.get('max_evidence_chars', 3600)
    generation_checks = []
    def annotate(messages):
        return _with_question_authority(_with_task_constraints(
            _with_task_background(messages,task_goal),task_constraints),anchor,question_interpretation,turn_message)
    def evaluate_model(messages):
        return evaluate_raw(annotate(messages))
    def evaluate(goal, evidence):
        return evaluate_requirements(goal, evidence, generate=evaluate_model,
            include_user_reply=True, context=context, final_k=final_k, max_evidence_chars=max_chars,
            repair_unselected_once=True, repair_finding_support_once=True, repair_evidence_quote_once=True,
            repair_response_shape_once=True)
    def generate(goal, evidence):
        messages = annotate(build_conversation_answer_messages(goal,evidence,context))
        for attempt in range(2):
            # Only an obtained but invalid answer can be repaired. Provider
            # failures/timeouts propagate immediately, including during repair.
            raw = generate_raw(messages)
            check = {'attempt': attempt + 1, 'raw_text': raw if isinstance(raw, str) else None,
                     'validation_error': None}
            generation_checks.append(check)
            try:
                # Preserve raw in generation_checks; formatting around an
                # exact, supplied ID does not require another model call.
                return validate_answer(raw, evidence, normalize_citation_format=True)
            except ValueError as error:
                check['validation_error'] = str(error)
                if attempt or not isinstance(raw, str):
                    raise
                details = {'validation_error': str(error),
                           'allowed_citation_ids': [e['chunk_id'] for e in evidence]}
                messages = [*messages, {'role': 'assistant', 'content': raw},
                    {'role': 'user', 'content': '上次输出未通过格式/引用校验：'
                     + json.dumps(details, ensure_ascii=False)
                     + '。请重新输出完整JSON对象，必须包含status、answer、citation_ids、missing_information。'
                       '引用只能逐字使用上述ID，正文方括号引用与citation_ids一致；不能省略字段或使用中文标点替代JSON语法。'
                       '保持原问题、本轮分析范围、环境和同一份证据；不要新增现场事实、资料或为通过检查强改为answer。'}]
    retrieval_question=question_interpretation if question_interpretation is not None else question
    query = build_initial_query(retrieval_question, context)
    result = run_rag_service(question, retrieve=retrieve, evaluate=evaluate, generate=generate,
                             initial_query=query, **service_options)
    result['context_input_version'] = context.input_version
    result['initial_query'] = query
    result['generation_checks'] = generation_checks
    if anchor is not None:
        result['question_anchor']=anchor.to_dict()
        result['retrieval_question']=retrieval_question
    if isinstance(context, AnalysisView):
        result['analysis_scope'] = context.scope.to_dict()
    if task_constraints is not None:
        result['task_constraints'] = task_constraints
    return result


def run_prepared_snapshot(snapshot: dict, *, retrieve, evaluate_raw, generate_raw, **service_options) -> dict:
    """Execute a trusted repository snapshot, using turn.question, never task.goal.

    The caller must read it through the authorized store and bind retrieval to
    the run's build. This function is not an HTTP/authentication layer or an
    automatic task selector; the store handles publication after execution.
    """
    if not isinstance(snapshot, dict) or set(snapshot) != {'task', 'turn', 'context'}:
        raise ValueError('require_prepared_turn_snapshot')
    task, turn = snapshot['task'], snapshot['turn']
    if (not isinstance(task, dict) or not isinstance(turn, dict)
            or any(not isinstance(task.get(k), str) or not task[k].strip() for k in ('task_id', 'goal'))
            or any(not isinstance(turn.get(k), str) or not turn[k].strip() for k in ('message_id', 'question'))
            or type(turn.get('input_version')) is not int):
        raise ValueError('invalid_prepared_turn')
    context = restore_context(snapshot['context'])
    if context.input_version != turn['input_version']:
        raise ValueError('turn_context_version_mismatch')
    if 'task_goal' in service_options:
        raise ValueError('prepared_task_background_cannot_be_overridden')
    if {'analysis_scope', 'allowed_targets'} & service_options.keys():
        raise ValueError('prepared_analysis_scope_cannot_be_overridden')
    if 'task_constraints' in service_options:
        raise ValueError('prepared_constraints_cannot_be_overridden')
    if {'question_anchor','question_interpretation','turn_message'} & service_options.keys():
        raise ValueError('prepared_question_authority_cannot_be_overridden')
    analysis_options = {}
    if 'analysis_scope' in turn:
        scope = restore_analysis_scope(turn['analysis_scope'])
        registry = task.get('registered_targets')
        if not isinstance(registry, list):
            raise ValueError('require_analysis_target_registry')
        try:
            targets = frozenset(FactTarget(**item['target']) for item in registry)
        except (TypeError, KeyError):
            raise ValueError('require_analysis_target_registry') from None
        if len(targets) != len(registry):
            raise ValueError('require_analysis_target_registry')
        analysis_options = {'analysis_scope': scope, 'allowed_targets': targets}
    if 'task_constraints' in turn:
        analysis_options['task_constraints'] = _validate_task_constraints(
            turn['task_constraints'],turn['input_version'],task['task_id'])
    if 'question_anchor' in turn:
        anchor=restore_question_anchor(turn['question_anchor'],task_id=task['task_id'],
                                       max_input_version=turn['input_version'])
        if turn['question']!=anchor.text:
            raise ValueError('question_does_not_match_original_source')
        analysis_options.update(question_anchor=anchor.to_dict(),
            question_interpretation=turn.get('retrieval_question'),
            turn_message={'message_id':turn['message_id'],'input_version':turn['input_version'],
                          'content':turn.get('raw_message')})
    elif 'retrieval_question' in turn:
        raise ValueError('retrieval_hint_requires_question_source')
    result = run_contextual_rag(turn['question'], context, task_goal=task['goal'],
        retrieve=retrieve, evaluate_raw=evaluate_raw, generate_raw=generate_raw,
        **analysis_options, **service_options)
    result.update(task_id=task['task_id'], turn_question=turn['question'], turn_input_version=turn['input_version'])
    return result
