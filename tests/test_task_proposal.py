"""Structural checks with scripted model replies, not recognition accuracy."""
from copy import deepcopy
from dataclasses import asdict
import json
import unittest

from app.context_proposal import DialogueMessage, RegisteredTarget
from app.conversation_context import ContextState, FactTarget, SourceQuote, UserFact, UserMessage
from app.conversation_preparation import prepare_task_context
from app.task_proposal import build_task_messages, prepare_task_turn, validate_task_proposal


MIXED = '先回到数据库连接那个问题。我刚确认，应用其实直接运行在宿主机上，数据库还是Compose里的。现在连接地址应该怎么填？'
ADDRESS = '应用运行在宿主机、数据库运行在Compose中时，连接地址应该怎么填？'


def view(content=MIXED):
    return {'state': ContextState(3), 'next_message': UserMessage('M4', 4, content),
        'prepared_task': {'task_id': 'T2'}, 'pending_task_clarifications': [],
        'task_catalog': [
            {'task_id': 'T1', 'goal': '排查生产环境数据库连接失败',
             'latest_input_version': 2, 'latest_question': 'pool_pre_ping如何处理失效连接？'},
            {'task_id': 'T2', 'goal': '排查镜像构建失败',
             'latest_input_version': 3, 'latest_question': '构建时为什么下载依赖失败？'}]}


def proposal(content=MIXED, task='T1', question=ADDRESS):
    return {'selection': {'operation': 'select', 'task_id': task, 'source_quote': content},
            'turn_request': {'mode': 'question', 'question': question,
                             'source': {'message_id': 'M4', 'quote': content}}}


class TaskProposalTests(unittest.TestCase):
    def accept(self, candidate, preparation=None):
        return validate_task_proposal(json.dumps(candidate, ensure_ascii=False), preparation or view())

    def test_return_to_old_task_reads_its_goal_and_keeps_new_question(self):
        captured = []
        def generate(messages):
            captured.append(json.loads(messages[1]['content']))
            return json.dumps(proposal())
        selected = prepare_task_turn(view(), generate=generate)
        self.assertEqual(selected.task_goal, '排查生产环境数据库连接失败')
        self.assertEqual(selected.question, ADDRESS)
        self.assertEqual(captured[0]['current_task_id'], 'T2')
        self.assertEqual(len(captured[0]['existing_tasks']), 2)
        self.assertNotIn('expected', captured[0])

    def test_model_cannot_rewrite_existing_goal_or_select_unknown_task(self):
        candidate = proposal()
        candidate['selection']['goal'] = '排查镜像构建失败'
        with self.assertRaisesRegex(ValueError, 'invalid_task_selection'):
            self.accept(candidate)
        candidate = proposal(task='foreign-conversation-task')
        with self.assertRaisesRegex(ValueError, 'task_not_in_authorized_catalog'):
            self.accept(candidate)

    def test_supplement_uses_latest_question_of_selected_task(self):
        content = '回到数据库问题，数据库映射的是5433端口。'
        candidate = proposal(content)
        candidate['turn_request'] = {'mode': 'continue', 'from_input_version': 2}
        self.assertEqual(self.accept(candidate, view(content)).question, 'pool_pre_ping如何处理失效连接？')
        for wrong in (1, 3, 4, True):
            candidate['turn_request']['from_input_version'] = wrong
            with self.subTest(version=wrong), self.assertRaisesRegex(ValueError, 'latest_same_task'):
                self.accept(candidate, view(content))

    def test_question_quote_cannot_come_from_arbitrary_history(self):
        candidate = proposal()
        candidate['turn_request']['source'] = {'message_id': 'M1', 'quote': '排查数据库连接失败'}
        with self.assertRaisesRegex(ValueError, 'turn_question_source_not_authorized'):
            self.accept(candidate)

    def test_first_task_requires_explicit_question(self):
        preparation = view('为什么启动失败？')
        preparation.update(state=ContextState(), next_message=UserMessage('M1', 1, '为什么启动失败？'),
                           prepared_task=None, task_catalog=[])
        candidate = {'selection': {'operation': 'create', 'goal': '排查启动失败', 'source_quote': '为什么启动失败'},
                     'turn_request': {'mode': 'question', 'question': '为什么启动失败？',
                                      'source': {'message_id': 'M1', 'quote': '为什么启动失败'}}}
        self.assertEqual(self.accept(candidate, preparation).task_goal, '排查启动失败')
        candidate['turn_request'] = {'mode': 'continue', 'from_input_version': 0}
        with self.assertRaisesRegex(ValueError, 'latest_same_task'):
            self.accept(candidate, preparation)

    def test_ambiguous_task_has_no_executable_question(self):
        candidate = {'selection': {'operation': 'clarify', 'candidate_task_ids': ['T1', 'T2'],
            'source_quote': MIXED, 'question': '指的是哪个故障？'}, 'turn_request': None}
        selected = self.accept(candidate)
        self.assertIsNone(selected.question)
        self.assertIsNone(selected.task_goal)
        candidate['turn_request'] = proposal()['turn_request']
        with self.assertRaisesRegex(ValueError, 'ambiguous_task_has_no_turn_question'):
            self.accept(candidate)

    def test_resolution_authorizes_only_the_linked_original_message(self):
        preparation = view('我说的是生产环境数据库问题。')
        preparation['pending_task_clarifications'] = [{'input_version': 3, 'message_id': 'M3',
            'content': '它在宿主机上，地址怎么填？', 'candidate_task_ids': ['T1', 'T2'],
            'question': '指数据库问题还是构建问题？'}]
        candidate = {'selection': {'operation': 'resolve_task', 'task_id': 'T1', 'clarification_version': 3,
            'source_quote': '生产环境数据库问题'}, 'turn_request': {'mode': 'question',
            'question': '应用在宿主机上时如何配置数据库地址？',
            'source': {'message_id': 'M3', 'quote': '它在宿主机上，地址怎么填？'}}}
        selected = self.accept(candidate, preparation)
        self.assertEqual(selected.deferred_message.message_id, 'M3')
        candidate['selection']['clarification_version'] = 2
        with self.assertRaisesRegex(ValueError, 'task_clarification_not_pending'):
            self.accept(candidate, preparation)

    def test_failure_does_not_fall_back_to_current_task(self):
        def timed_out(messages):
            raise TimeoutError('model_timeout')
        with self.assertRaises(TimeoutError):
            prepare_task_turn(view(), generate=timed_out)
        for raw in ('not-json', '{}', '{"selection":[],"turn_request":null}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare_task_turn(view(), generate=lambda _: raw)

    def test_invalid_server_view_is_rejected_before_model(self):
        called = []
        preparation = view()
        preparation['task_catalog'].append(deepcopy(preparation['task_catalog'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate_task_catalog_id'):
            prepare_task_turn(preparation, generate=lambda _: called.append(True))
        self.assertEqual(called, [])
        with self.assertRaisesRegex(ValueError, 'duplicate_dialogue_message'):
            build_task_messages(view(), recent_dialogue=(DialogueMessage('M4', 'user', MIXED),))

    def test_context_updates_only_selected_scope_and_preserves_other_task(self):
        app = FactTarget('app', 'production', 'runtime_location')
        other = FactTarget('builder', 'build_environment', 'runtime_location')
        state = ContextState(3, (UserFact('F1', app, 'container', SourceQuote('M1', '应用在容器')),
                                UserFact('F2', other, 'container', SourceQuote('M3', '构建在容器'))))
        decision = self.accept(proposal())
        patch = {'updates': [{'operation': 'replace', 'target': asdict(app), 'previous_fact_id': 'F1',
            'value': 'host', 'source': {'message_id': 'M4', 'quote': '应用其实直接运行在宿主机上'}}]}
        captured = []
        def generate(messages):
            captured.append(json.loads(messages[1]['content']))
            return json.dumps(patch)
        result = prepare_task_context(state, view()['next_message'], decision,
            registry=(RegisteredTarget(app, '生产应用运行位置'),), generate=generate)
        self.assertEqual([f.fact_id for f in result.state.current_facts if f.target == other], ['F2'])
        self.assertEqual([f.value for f in result.state.current_facts if f.target == app], ['host'])
        self.assertEqual(len(captured[0]['current_user_stated_facts']), 1)
        self.assertEqual(captured[0]['selected_task']['goal'], decision.task_goal)
        patch['updates'][0].update(target=asdict(other), previous_fact_id='F2')
        with self.assertRaisesRegex(ValueError, 'unregistered_target'):
            prepare_task_context(state, view()['next_message'], decision,
                registry=(RegisteredTarget(app, '生产应用运行位置'),), generate=generate)

    def test_scoped_context_can_cite_resolved_original_but_not_other_history(self):
        preparation = view('我说的是生产环境数据库问题。')
        preparation['pending_task_clarifications'] = [{'input_version': 3, 'message_id': 'M3',
            'content': '应用在宿主机', 'candidate_task_ids': ['T1', 'T2'], 'question': '指哪个环境？'}]
        candidate = {'selection': {'operation': 'resolve_task', 'task_id': 'T1', 'clarification_version': 3,
            'source_quote': '生产环境数据库问题'}, 'turn_request': {'mode': 'continue', 'from_input_version': 2}}
        selected = self.accept(candidate, preparation)
        app = FactTarget('app', 'production', 'runtime_location')
        patch = {'updates': [{'operation': 'add', 'target': asdict(app), 'previous_fact_id': None,
            'value': 'host', 'source': {'message_id': 'M3', 'quote': '应用在宿主机'}}]}
        result = prepare_task_context(preparation['state'], preparation['next_message'], selected,
            registry=(RegisteredTarget(app, '应用运行位置'),), generate=lambda _: json.dumps(patch))
        self.assertEqual(result.state.current_facts[0].source.message_id, 'M3')
        patch['updates'][0]['source']['message_id'] = 'M1'
        with self.assertRaisesRegex(ValueError, 'source_quote_not_in_authorized_user_message'):
            prepare_task_context(preparation['state'], preparation['next_message'], selected,
                registry=(RegisteredTarget(app, '应用运行位置'),), generate=lambda _: json.dumps(patch))
