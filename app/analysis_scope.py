"""A frozen, conditional reasoning view; never a mutation of actual user state.

The server must authorize task/source references before supplying this scope.
This pure runtime module checks shape, field registration and version bounds;
it cannot establish that a quoted user statement is true or that an inheritance
reference belongs to an authorized task. It performs no SQL or model calls.
"""
from dataclasses import asdict, dataclass

from app.conversation_context import (
    ContextState, FactTarget, SourceQuote, build_reasoning_context, restore_context,
)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Assumption:
    target: FactTarget
    value: str
    source: SourceQuote

    def __post_init__(self):
        if (not isinstance(self.target, FactTarget) or not _text(self.value)
                or not isinstance(self.source, SourceQuote)
                or not _text(self.source.message_id) or not _text(self.source.quote)):
            raise ValueError('invalid_hypothetical_assumption')


@dataclass(frozen=True)
class AnalysisScope:
    mode: str
    assumptions: tuple[Assumption, ...] = ()
    inherited_from_input_version: int | None = None

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in ('current', 'hypothetical'):
            raise ValueError('analysis_scope_not_executable')
        if (not isinstance(self.assumptions, tuple)
                or any(not isinstance(a, Assumption) for a in self.assumptions)):
            raise ValueError('invalid_analysis_assumptions')
        if len({a.target for a in self.assumptions}) != len(self.assumptions):
            raise ValueError('duplicate_assumption_target')
        inherited = self.inherited_from_input_version
        if inherited is not None and (type(inherited) is not int or inherited < 1):
            raise ValueError('invalid_analysis_inheritance_version')
        if self.mode == 'current' and (self.assumptions or inherited is not None):
            raise ValueError('current_scope_cannot_contain_assumptions')
        if self.mode == 'hypothetical' and not self.assumptions:
            raise ValueError('hypothetical_scope_requires_assumptions')

    def to_dict(self):
        return {'mode': self.mode, 'assumptions': [asdict(a) for a in self.assumptions],
                'inherited_from_input_version': self.inherited_from_input_version}


def restore_analysis_scope(raw: dict) -> AnalysisScope:
    """Read a trusted frozen scope, not an arbitrary client-supplied proposal."""
    if (not isinstance(raw, dict)
            or set(raw) != {'mode', 'assumptions', 'inherited_from_input_version'}
            or not isinstance(raw['assumptions'], list)):
        raise ValueError('invalid_analysis_scope_snapshot')
    assumptions = []
    for item in raw['assumptions']:
        if (not isinstance(item, dict) or set(item) != {'target', 'value', 'source'}
                or not isinstance(item['target'], dict)
                or set(item['target']) != {'entity_id', 'scope_id', 'field'}
                or not isinstance(item['source'], dict)
                or set(item['source']) != {'message_id', 'quote'}):
            raise ValueError('invalid_hypothetical_assumption')
        assumptions.append(Assumption(FactTarget(**item['target']), item['value'], SourceQuote(**item['source'])))
    return AnalysisScope(raw['mode'], tuple(assumptions), raw['inherited_from_input_version'])


@dataclass(frozen=True)
class AnalysisView:
    actual_context: ContextState
    scope: AnalysisScope

    @property
    def input_version(self):
        return self.actual_context.input_version

    @property
    def assumed_targets(self):
        return frozenset(a.target for a in self.scope.assumptions)

    @property
    def baseline_facts(self):
        # No invented UserFact: assumptions remain a separate, labelled type.
        return tuple(f for f in self.actual_context.current_facts if f.target not in self.assumed_targets)

    def to_payload(self):
        payload = build_reasoning_context(self.actual_context)
        payload['user_stated_facts'] = [item for item in payload['user_stated_facts']
            if FactTarget(**item['target']) not in self.assumed_targets]
        for item in payload['pending_clarifications']:
            item['assumed_for_this_turn'] = FactTarget(**item['target']) in self.assumed_targets
        payload['analysis_scope'] = self.scope.to_dict()
        return payload


def build_analysis_view(context: ContextState, scope: AnalysisScope, *,
                        allowed_targets: frozenset[FactTarget]) -> AnalysisView:
    if (not isinstance(context, ContextState) or type(context.input_version) is not int
            or context.input_version < 1 or not isinstance(scope, AnalysisScope)):
        raise ValueError('require_prepared_analysis_context')
    if (not isinstance(allowed_targets, frozenset)
            or any(not isinstance(t, FactTarget) for t in allowed_targets)):
        raise ValueError('require_analysis_target_registry')
    if any(item.target not in allowed_targets for item in (
            *context.current_facts, *context.pending_facts, *scope.assumptions)):
        raise ValueError('analysis_target_not_registered')
    inherited = scope.inherited_from_input_version
    if inherited is not None and inherited >= context.input_version:
        raise ValueError('analysis_inheritance_must_precede_turn')
    # Also validate the context representation. Keep old values only in this
    # private snapshot, never in the model's current facts or retrieval query.
    return AnalysisView(restore_context(context.to_dict()), scope)


def project_reasoning_context(context: ContextState | AnalysisView) -> dict:
    if isinstance(context, AnalysisView):
        return context.to_payload()
    if (not isinstance(context, ContextState) or type(context.input_version) is not int
            or context.input_version < 1):
        raise ValueError('require_prepared_conversation_context')
    return build_reasoning_context(context)


ANALYSIS_SCOPE_PROMPT = '''
本轮有明确的analysis_scope。mode=current时按当前用户陈述分析；mode=hypothetical时，
assumptions是用户要求用于本轮推理的假设，不是已经发生的变更或独立核验的事实。
对于被假设覆盖的同一target，按其假设值分析；不能用实际环境的旧值或待澄清状态
否定该条件问题。其余user_stated_facts仅为沿用的当前背景，不保证未来部署必然不变。
pending_clarifications仍保存实际未知项：assumed_for_this_turn=true时，该字段的未知
不阻塞本轮纯条件问题，pending_impacts中对应blocking_requirement_ids必须为[]。
其他未知项是否影响本轮仍须逐需求判断，不能因为有假设就忽略所有信息缺口。
假设只提供条件，不能代替文档证据，也不能据此推断真实故障根因或已修复。
公开结论必须说明假设条件和适用范围；本轮question仍决定要回答的内容，不重跑旧问题。
'''
