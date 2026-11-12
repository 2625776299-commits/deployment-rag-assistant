"""Resolve a proposed analysis scope from canonical, task-scoped server data.

Called before a commit for early feedback and again inside the SQL transaction.
Quote checks establish provenance, not the meaning or truth of a user statement.
"""
from app.analysis_scope import AnalysisScope, Assumption, restore_analysis_scope
from app.conversation_context import FactTarget, SourceQuote


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _source(raw, messages):
    if (not isinstance(raw, dict) or set(raw) != {'message_id', 'quote'}
            or not _text(raw['message_id']) or raw['message_id'] not in messages
            or not _text(raw['quote']) or raw['quote'] not in messages[raw['message_id']].content):
        raise ValueError('analysis_source_not_authorized')
    return SourceQuote(**raw)


def _previous(history, version, message, *, require_hypothetical=True):
    if type(version) is not int or not 0 < version < message.input_version:
        raise ValueError('invalid_analysis_inheritance_version')
    row = next((r for r in history if r['input_version'] == version), None)
    if row is None:
        raise ValueError('analysis_reference_not_in_selected_task')
    scope = restore_analysis_scope(row['analysis_scope'])
    if require_hypothetical and scope.mode != 'hypothetical':
        raise ValueError('analysis_reference_is_not_hypothetical')
    return scope


def resolve_analysis_request(request, message, *, allowed_targets, history=(), deferred_message=None):
    """history must be read by the server for the selected task and owner."""
    fields = {
        'current': {'mode', 'source', 'confirmation_from_input_version'},
        'hypothetical': {'mode', 'source', 'assumptions'},
        'inherit': {'mode', 'source', 'from_input_version', 'assumption_updates'},
        'clarify': {'mode', 'source', 'question'},
    }
    if not isinstance(request, dict):
        raise ValueError('require_analysis_request')
    mode = request.get('mode')
    if not isinstance(mode, str) or mode not in fields or set(request) != fields[mode]:
        raise ValueError('invalid_analysis_request')
    messages = {message.message_id: message}
    if deferred_message is not None:
        messages[deferred_message.message_id] = deferred_message
    _source(request['source'], messages)
    if mode == 'current':
        version = request['confirmation_from_input_version']
        if version is not None:
            _previous(history, version, message)
        return AnalysisScope('current').to_dict()
    if mode == 'clarify':
        if not _text(request['question']):
            raise ValueError('invalid_analysis_clarification')
        return {'mode': 'needs_clarification', 'question': request['question'], 'source': dict(request['source'])}
    if mode == 'hypothetical':
        scope = restore_analysis_scope({'mode': mode, 'assumptions': request['assumptions'],
                                        'inherited_from_input_version': None})
        for a in scope.assumptions:
            _source({'message_id': a.source.message_id, 'quote': a.source.quote}, messages)
        # A complete repeated condition set is an explicit version reference,
        # not a newly sourced hypothesis. Reject rather than guess a history
        # row; the existing bounded correction must select an authorized version.
        conditions = {a.target: a.value for a in scope.assumptions}
        for row in history:
            if not 0 < row['input_version'] < message.input_version:
                continue
            prior = row['analysis_scope']
            if prior.get('mode') != 'hypothetical':
                continue
            previous = restore_analysis_scope(prior)
            if conditions == {a.target: a.value for a in previous.assumptions}:
                raise ValueError('repeated_hypothesis_requires_inheritance')
    else:
        previous = _previous(history, request['from_input_version'], message, require_hypothetical=False)
        updates = request['assumption_updates']
        if not isinstance(updates, list):
            raise ValueError('invalid_assumption_updates')
        if previous.mode == 'current':
            if updates:
                raise ValueError('current_scope_inheritance_cannot_modify_assumptions')
            # Inherit the mode, not an old environment snapshot. The original
            # reference remains in analysis_request for audit.
            return AnalysisScope('current').to_dict()
        merged = {a.target: a for a in previous.assumptions}
        seen = set()
        for update in updates:
            if (not isinstance(update, dict) or set(update) != {'target', 'value', 'source'}
                    or not isinstance(update['target'], dict)
                    or set(update['target']) != {'entity_id', 'scope_id', 'field'}):
                raise ValueError('invalid_assumption_update')
            target = FactTarget(**update['target'])
            if target in seen:
                raise ValueError('duplicate_assumption_target')
            seen.add(target)
            source = _source(update['source'], messages)
            if update['value'] is None:
                if target not in merged:
                    raise ValueError('cannot_remove_unknown_assumption')
                del merged[target]
            else:
                # Reaffirmations remain in analysis_request for audit; they do
                # not replace the earliest source of the unchanged condition.
                if target not in merged or merged[target].value != update['value']:
                    merged[target] = Assumption(target, update['value'], source)
        scope = AnalysisScope('hypothetical', tuple(merged.values()), request['from_input_version'])
    if any(a.target not in allowed_targets for a in scope.assumptions):
        raise ValueError('analysis_target_not_registered')
    return scope.to_dict()


def guard_hypothetical_context_sources(proposal, resolved_scope):
    """Do not reuse the same hypothetical clause to assert a real change.

Separate actual and hypothetical clauses about the same field remain possible.
This catches source overlap, not all semantic mistakes in model extraction.
"""
    if resolved_scope is None or resolved_scope['mode'] != 'hypothetical':
        return
    scope = restore_analysis_scope(resolved_scope)
    for update in proposal['updates']:
        if not isinstance(update, dict) or not isinstance(update.get('source'), dict):
            raise ValueError('invalid_update_proposal')
        for a in scope.assumptions:
            source = update.get('source', {})
            if (source.get('message_id') == a.source.message_id
                    and _text(source.get('quote'))
                    and (source['quote'] in a.source.quote or a.source.quote in source['quote'])):
                raise ValueError('hypothetical_quote_cannot_update_actual_context')


ANALYSIS_PREPARATION_PROMPT = '''
本入口同时提议环境更新和本轮分析范围。只输出一个对象：
{"updates":[...],"analysis_request":{...}}。updates仍严格遵循上面的操作格式。
analysis_history由服务端按本次选中的任务读取，包含历史范围、版本与问题。
分析条件与本轮问题分开：新问题可继承旧假设，不等于继续回答旧问题。

analysis_request精确字段：
mode=current时，analysis_request本身为{"mode":"current","source":S,"confirmation_from_input_version":null}
mode=hypothetical时，analysis_request本身为{"mode":"hypothetical","source":S,"assumptions":[A,...]}
mode=inherit时，analysis_request本身为{"mode":"inherit","source":S,"from_input_version":历史版本整数,"assumption_updates":[]}
mode=clarify时，analysis_request本身为{"mode":"clarify","source":S,"question":"你问的是实际部署还是刚才的假设？"}
analysis_request直接包含mode等字段，不能再包一层current/hypothetical/inherit/clarify键。
完整结构示例（字段中的占位原文必须替换为本轮真实ID和原文）：
{"updates":[],"analysis_request":{"mode":"current","source":{"message_id":"本轮ID","quote":"本轮原话"},"confirmation_from_input_version":null}}
A精确为{"target":T,"value":"假定值","source":S}，T完整复制registered_targets。
S是当前用户消息的ID和连续原文；已解决归属时也可用resolved_original_message。
新假设的来源只用这些原话，不自造引文；继承时不要重写旧条件与来源，由程序读取。
与同任务历史某一套条件完全相同时必须用inherit并指定from_input_version，
不能把“刚才的假设”等指代当成这些条件的新来源。即使本轮完整重述相同条件，
也用inherit保存本轮承接依据；无法确定承接哪个版本时clarify，不猜最近版本。
继承后不变的条件保留原来源；明确改动的条件才记录新原文，未提及的不自动删除。
inherit的assumption_updates可用同一A结构改动个别条件；value=null表示明确取消该条件。
inherit也可承接同任务历史的current模式，此时assumption_updates必须为[]，只沿用
实际分析模式，环境字段仍读取本轮最新状态，不回滚成历史字段；新假设须用hypothetical。

用户问当前部署、通用机制或明确回到实际时用current。只有本轮确实要求条件分析或
明确承接假设时用hypothetical/inherit；依据同任务相关问题，不机械继承最近消息。
后续省略问题可承接明确的假设讨论；有多种合理解释且会影响结论时用clarify。
范围不明不是任务归属不明，不创建新任务。实际未知不等于范围不明。
纯假设不改当前环境、不让旧事实变待澄清。实际与假设可以同句出现，updates与
assumptions的source分别引用对应的实际/假设子句，不将整段混合原文作为两者共同依据。
实际未知但本轮已经明确假设其值时，保留实际pending，不为条件问题重复追问。

用户说“已按刚才方案完成”，所指历史方案唯一且范围清楚时，可以更新对应用户陈述，
不必实测；current的confirmation_from_input_version引用被确认的hypothetical历史版本，
updates依据本轮确认原话，不能单独用旧假设作实施证明。不得据此声称已独立核验、
已连通、故障已修复。仅说“迁入Docker”不能推出同网，方案不清楚则针对性澄清。
来源/字段/引用校验不证明语义正确；不要从助手建议、资料示例或默认值补现场事实。
'''
