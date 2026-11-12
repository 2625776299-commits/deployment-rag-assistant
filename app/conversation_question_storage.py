"""Resolve question provenance under the authorized preparation transaction."""
from psycopg import sql

from app.conversation_context import UserMessage
from app.question_authority import anchor_from_message, restore_question_anchor


class ConversationQuestionStorage:
    def _question_anchor(self, conn, conversation_id, task_id, message, request, deferred_message=None):
        if task_id is None:
            return None
        if request['mode']=='question':
            authorized={message.message_id:message}
            if deferred_message is not None:
                authorized[deferred_message.message_id]=deferred_message
            canonical=authorized.get(request['source']['message_id'])
            if canonical is None:
                raise ValueError('question_anchor_source_not_authorized')
            return anchor_from_message(task_id,canonical,request['source'],message.input_version).to_dict()
        # The question continuation validator already checked latest same-task.
        # Legacy rows can be followed using canonical requests/messages; do not
        # rewrite their old rows or invent a source from a generated question.
        version=request['from_input_version']
        ceiling=message.input_version
        while True:
            if type(version) is not int or not 0 < version < ceiling:
                raise ValueError('invalid_question_anchor_lineage')
            row=conn.execute(sql.SQL('''SELECT task_id,question_anchor,turn_request,selection FROM {}
                WHERE conversation_id=%s AND input_version=%s''').format(self._table('preparations')),
                (conversation_id,version)).fetchone()
            if row is None or row[0] != task_id:
                raise ValueError('question_anchor_not_in_selected_task')
            if row[1] is not None:
                return restore_question_anchor(row[1],task_id=task_id,max_input_version=version).to_dict()
            previous=row[2]
            if not isinstance(previous,dict):
                raise ValueError('legacy_question_requires_explicit_source')
            if previous.get('mode')=='continue':
                ceiling,version=version,previous.get('from_input_version')
                continue
            if previous.get('mode')!='question' or not isinstance(previous.get('source'),dict):
                raise ValueError('legacy_question_requires_explicit_source')
            source=previous['source']
            canonical_row=conn.execute(sql.SQL('''SELECT message_id,input_version,content FROM {}
                WHERE conversation_id=%s AND message_id=%s''').format(self._table('messages')),
                (conversation_id,source.get('message_id'))).fetchone()
            if canonical_row is None:
                raise ValueError('question_anchor_source_not_authorized')
            canonical=UserMessage(*canonical_row)
            if canonical.input_version != version:
                selection=row[3]
                if (selection.get('operation')!='resolve_task'
                        or selection.get('clarification_version') != canonical.input_version):
                    raise ValueError('question_anchor_source_not_authorized')
                resolved=conn.execute(sql.SQL('''SELECT 1 FROM {} WHERE conversation_id=%s
                    AND clarification_version=%s AND resolved_input_version=%s''')
                    .format(self._table('clarification_resolutions')),
                    (conversation_id,canonical.input_version,version)).fetchone()
                original=self._preparation(conn,conversation_id,canonical.input_version)
                if not resolved or original is None or task_id not in original[1].get('candidate_task_ids',[]):
                    raise ValueError('question_anchor_source_not_authorized')
            return anchor_from_message(task_id,canonical,source,version).to_dict()
