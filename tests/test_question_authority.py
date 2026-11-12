"""Original demand versus a deliberately misleading retrieval interpretation."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from app.conversation_context import ContextState,UserMessage
from app.conversation_rag import run_prepared_snapshot
from app.question_authority import anchor_from_message,restore_question_anchor


ORIGINAL='现在能排除其他原因，认定这次数据库连接失败就是密码错误吗？'
BAD_HINT='在已经排除其他原因的前提下，确认本次连接失败是密码错误。'
RULE={'chunk_id':'Q:1','content':'服务名由内部DNS解析。',
      'metadata':{'document_id':'fixture','section_id':'Q','source_url':'https://example.invalid/q'}}


def snapshot(text=ORIGINAL,hint=BAD_HINT,version=1,mid='M1',raw=None):
    anchor=anchor_from_message('T1',UserMessage('M1',1,text),{'message_id':'M1','quote':text},1)
    return {'task':{'task_id':'T1','goal':'诊断连接失败'},'context':ContextState(version).to_dict(),
        'turn':{'message_id':mid,'input_version':version,'raw_message':raw or text,
                'question':text,'retrieval_question':hint,'question_anchor':anchor.to_dict()}}


def proposal(question,action='insufficient'):
    return json.dumps({'requirements':[{'requirement_id':'R1','requirement':question,'question_quote':question}],
        'assessments':[{'requirement_id':'R1','coverage':'covered' if action=='answer' else 'missing',
            'evidence':[{'chunk_id':'Q:1','quote':RULE['content']}] if action=='answer' else [],
            'gap':None if action=='answer' else '没有本次现场因果证据'}],
        'action':action,'selected_chunk_ids':['Q:1'] if action=='answer' else [],
        'missing_knowledge':['无法证明密码错误或排除其他原因'] if action=='insufficient' else [],
        'missing_user_information':[],'search_query':None,'reason':'scripted contract example',
        'known_findings':[],'pending_impacts':[]})


class QuestionAuthorityTests(unittest.TestCase):
    def test_bad_interpretation_is_only_retrieval_hint_and_not_evaluation_question(self):
        received=[];retrieve=Mock(return_value=[RULE]);generate=Mock()
        def evaluate(messages):
            received.append(json.loads(messages[1]['content']))
            return proposal(ORIGINAL)
        s=snapshot();before=deepcopy(s)
        result=run_prepared_snapshot(s,retrieve=retrieve,evaluate_raw=evaluate,generate_raw=generate)
        self.assertEqual(result['status'],'insufficient')
        self.assertTrue(retrieve.call_args.args[0].startswith(BAD_HINT))
        self.assertEqual(received[0]['question'],ORIGINAL)
        self.assertEqual(received[0]['question_interpretation'],BAD_HINT)
        self.assertEqual(received[0]['question_source']['message_id'],'M1')
        self.assertEqual(s,before);generate.assert_not_called()

    def test_supplement_does_not_replace_original_question_and_reaches_both_consumers(self):
        original='先解释服务名发现的一般机制。';supplement='这次日志拿不到，但还是只问一般机制。'
        s=snapshot(original,'解释DNS服务发现',2,'M2',supplement)
        received=[]
        def evaluate(messages):
            received.append(json.loads(messages[1]['content']));return proposal(original,'answer')
        def generate(messages):
            received.append(json.loads(messages[1]['content']))
            return json.dumps({'status':'answer','answer':'内部DNS解析服务名。[Q:1]',
                               'citation_ids':['Q:1'],'missing_information':[]})
        result=run_prepared_snapshot(s,retrieve=lambda _:[RULE],evaluate_raw=evaluate,generate_raw=generate)
        self.assertEqual(result['status'],'answer')
        for p in received:
            self.assertEqual(p['question'],original)
            self.assertEqual(p['current_user_message']['content'],supplement)
            self.assertEqual(p['question_source']['message_id'],'M1')

    def test_requirement_cannot_quote_the_interpretation_as_original_user_question(self):
        generate=Mock()
        result=run_prepared_snapshot(snapshot(),retrieve=lambda _:[RULE],
            evaluate_raw=lambda _:proposal(BAD_HINT),generate_raw=generate)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error']['code'],'requirement_quote_not_in_question')
        generate.assert_not_called()

    def test_forged_task_future_source_and_replaced_original_stop_before_retrieval(self):
        for mutate,error in ((lambda s:s['task'].update(task_id='T2'),'question_anchor_task_mismatch'),
            (lambda s:s['turn']['question_anchor'].update(introduced_in_input_version=2),'question_anchor_from_future'),
            (lambda s:s['turn'].update(question=BAD_HINT),'question_does_not_match_original_source'),
            (lambda s:s['turn'].update(raw_message='不同原话'),'question_source_conflicts_with_current_message')):
            s=snapshot();mutate(s);retrieve=Mock()
            with self.subTest(error=error),self.assertRaisesRegex(ValueError,error):
                run_prepared_snapshot(s,retrieve=retrieve,evaluate_raw=Mock(),generate_raw=Mock())
            retrieve.assert_not_called()

    def test_full_raw_text_is_retained_even_when_model_source_quote_is_narrow(self):
        text='先不要下结论。\n  yaml: true\n能否排除其他原因？'
        a=anchor_from_message('T1',UserMessage('M1',1,text),
                              {'message_id':'M1','quote':'排除其他原因'},1)
        self.assertEqual(a.text,text)
        self.assertEqual(restore_question_anchor(json.loads(json.dumps(a.to_dict()))),a)

