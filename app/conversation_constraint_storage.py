"""Constraint storage helpers. Caller owns the authorized preparation transaction."""
from psycopg import sql

from app.task_constraints import ConstraintSnapshot, restore_constraint_snapshot
from app.constraint_preparation import resolve_evidence_request


class ConversationConstraintStorage:
    def _initialize_constraints(self, conn):
        for name in ('evidence_request','incident_scope','constraint_snapshot'):
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} JSONB')
                .format(self._table('preparations'),sql.Identifier(name)))
        conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
            conversation_id TEXT NOT NULL, task_id TEXT NOT NULL, incident_id TEXT NOT NULL,
            description TEXT NOT NULL CHECK(length(trim(description))>0),
            source_message_id TEXT NOT NULL, source_quote TEXT NOT NULL,
            registered_input_version BIGINT NOT NULL CHECK(registered_input_version>0),
            PRIMARY KEY(conversation_id,task_id,incident_id),
            FOREIGN KEY(conversation_id,task_id) REFERENCES {}(conversation_id,task_id),
            FOREIGN KEY(conversation_id,source_message_id) REFERENCES {}(conversation_id,message_id),
            FOREIGN KEY(conversation_id,registered_input_version) REFERENCES {}(conversation_id,input_version)
        )''').format(self._table('incidents'),self._table('tasks'),self._table('messages'),self._table('messages')))

    def _constraint_view(self, conn, conversation_id, base_version):
        row = conn.execute(sql.SQL('''SELECT constraint_snapshot FROM {}
            WHERE conversation_id=%s AND input_version<=%s AND constraint_snapshot IS NOT NULL
            ORDER BY input_version DESC LIMIT 1''').format(self._table('preparations')),
            (conversation_id,base_version)).fetchone()
        old = restore_constraint_snapshot(row[0]) if row else ConstraintSnapshot()
        if old.input_version > base_version:
            raise ValueError('constraint_snapshot_version_mismatch')
        state = ConstraintSnapshot(base_version,old.current)
        rows = conn.execute(sql.SQL('''SELECT i.task_id,i.incident_id,i.description,
            i.source_message_id,i.source_quote,i.registered_input_version,
            (SELECT max(p.input_version) FROM {} p WHERE p.conversation_id=i.conversation_id
               AND p.task_id=i.task_id AND p.input_version<=%s
               AND p.incident_scope->>'incident_id'=i.incident_id)
            FROM {} i JOIN {} m ON m.conversation_id=i.conversation_id AND m.message_id=i.source_message_id
            WHERE i.conversation_id=%s AND i.registered_input_version<=%s ORDER BY i.registered_input_version,i.incident_id''')
            .format(self._table('preparations'),self._table('incidents'),self._table('messages')),
            (base_version,conversation_id,base_version)).fetchall()
        catalog = [dict(zip(('task_id','incident_id','description','source_message_id','source_quote',
                            'introduced_in_input_version','latest_selected_input_version'),r)) for r in rows]
        pending = conn.execute(sql.SQL('''SELECT p.task_id,p.input_version,m.message_id,m.content,p.incident_scope
            FROM {} p JOIN {} m ON m.conversation_id=p.conversation_id AND m.input_version=p.input_version
            WHERE p.conversation_id=%s AND p.input_version<=%s
              AND p.incident_scope->>'mode'='needs_clarification'
              AND NOT EXISTS(SELECT 1 FROM {} later WHERE later.conversation_id=p.conversation_id
                 AND later.task_id=p.task_id AND later.input_version<=%s
                 AND later.evidence_request->>'resolves_input_version'=p.input_version::text)
            ORDER BY p.input_version''').format(self._table('preparations'),self._table('messages'),
                                               self._table('preparations')),
            (conversation_id,base_version,base_version)).fetchall()
        return {'constraint_state':state,'incident_catalog':catalog,
                'pending_incidents':[dict(task_id=t,input_version=v,message_id=mid,content=text,
                    question=scope['question'],candidate_incident_ids=scope['candidate_incident_ids'])
                    for t,v,mid,text,scope in pending]}

    def _commit_evidence_request(self, conn, conversation_id, task_id, message, base_version,
                                 request, deferred_message=None):
        view = self._constraint_view(conn,conversation_id,base_version)
        prepared = resolve_evidence_request(request,message,task_id=task_id,
            state=view['constraint_state'],catalog=view['incident_catalog'],
            pending=view['pending_incidents'],deferred_message=deferred_message)
        # Deferred attribution cannot override a newer explicit confirmation,
        # including one that kept the same value and original record identity.
        if prepared.scope.get('mode') == 'incident':
            iid = prepared.scope['incident_id']
            for update in request['updates']:
                source_id = update['source']['message_id']
                if source_id == message.message_id:
                    continue
                source_row = conn.execute(sql.SQL('SELECT input_version FROM {} WHERE conversation_id=%s AND message_id=%s')
                    .format(self._table('messages')),(conversation_id,source_id)).fetchone()
                rows = conn.execute(sql.SQL('''SELECT evidence_request FROM {} WHERE conversation_id=%s
                    AND task_id=%s AND input_version>%s AND input_version<=%s
                    AND incident_scope->>'incident_id'=%s AND evidence_request IS NOT NULL''')
                    .format(self._table('preparations')),
                    (conversation_id,task_id,source_row[0],base_version,iid)).fetchall()
                old = next((c for c in view['constraint_state'].current if c.incident.task_id==task_id
                    and c.incident.incident_id==iid and c.evidence_kind==update['evidence_kind']),None)
                if (old is not None and old.availability != update['availability']
                        and any(u['evidence_kind']==update['evidence_kind'] for r in rows for u in r[0]['updates'])):
                    raise ValueError('deferred_constraint_would_overwrite_newer_record')
        if prepared.registration is not None:
            r = prepared.registration
            conn.execute(sql.SQL('''INSERT INTO {} (conversation_id,task_id,incident_id,description,
                source_message_id,source_quote,registered_input_version) VALUES (%s,%s,%s,%s,%s,%s,%s)''').format(self._table('incidents')),
                (conversation_id,task_id,r['incident_id'],r['description'],r['source_message_id'],r['source_quote'],message.input_version))
        return prepared.scope,prepared.state.to_dict()
