"""An immutable question source, distinct from a model's retrieval interpretation.

The authorized store supplies canonical messages. Quote membership and version
checks establish provenance, not semantic task selection or question completeness.
"""
from dataclasses import asdict, dataclass

from app.conversation_context import UserMessage


def _text(value):
    return isinstance(value,str) and bool(value.strip())


@dataclass(frozen=True)
class QuestionAnchor:
    task_id: str
    message_id: str
    message_input_version: int
    text: str
    source_quote: str
    introduced_in_input_version: int

    def __post_init__(self):
        if (not all(_text(v) for v in (self.task_id,self.message_id,self.text,self.source_quote))
                or self.source_quote not in self.text
                or type(self.message_input_version) is not int
                or type(self.introduced_in_input_version) is not int
                or not 0 < self.message_input_version <= self.introduced_in_input_version):
            raise ValueError('invalid_question_anchor')

    def to_dict(self):
        return asdict(self)


def anchor_from_message(task_id, message, source, introduced_version):
    if (not isinstance(message,UserMessage) or not isinstance(source,dict)
            or set(source) != {'message_id','quote'} or source['message_id'] != message.message_id):
        raise ValueError('question_anchor_source_not_authorized')
    return QuestionAnchor(task_id,message.message_id,message.input_version,
                          message.content,source['quote'],introduced_version)


def restore_question_anchor(raw, *, task_id=None, max_input_version=None):
    if (not isinstance(raw,dict) or set(raw) != {'task_id','message_id','message_input_version',
            'text','source_quote','introduced_in_input_version'}):
        raise ValueError('invalid_question_anchor')
    anchor=QuestionAnchor(**raw)
    if task_id is not None and anchor.task_id != task_id:
        raise ValueError('question_anchor_task_mismatch')
    if max_input_version is not None and (type(max_input_version) is not int
            or anchor.introduced_in_input_version > max_input_version):
        raise ValueError('question_anchor_from_future')
    return anchor


QUESTION_AUTHORITY_PROMPT = '''
question是服务器从question_source对应用户原消息逐字保存的提问依据。
question_interpretation（如有）只是模型对指代/检索的解释提议，不是新事实，
不能替换原问题的疑问、否定、条件或不确定性：询问“能否排除其他原因”
不能变成“已排除其他原因”的前提。需求引用question中的原文，不引用改写当原话。
原始消息可以同时含部署陈述和问题：环境字段仍以当前conversation_context、
task_constraints及analysis_scope为准；原问题中的旧部署值不恢复成当前事实。
current_user_message（如有）是本轮补充/归属确认，与原提问分开保留；
继续旧问题时保留其请求目标，结合本轮明确更正、范围限制与准备后的状态处理，
不能把“日志没了”等补充本身当作新问题，也不能忽略当前明确的限制。
任务归属限定要处理的对象；其他任务的内容不能挪入本任务。
用当前上下文、分析条件和解释提议消解原话中的指代，但不得新增未成立的前提。
如果仍有多种影响结论的解释，应说明需要澄清，不擅自选择一种或按助手猜测作答。
这些输入约定不保证语义一定理解正确，资料支持和需求覆盖仍须分别核对。
'''
