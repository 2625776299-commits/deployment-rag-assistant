"""PostgreSQL conversation history and version-guarded result publication.

Trusted server code supplies owner_id, registered targets and proposed updates.
This repository does not authenticate users, extract facts, invoke models, or
expose HTTP routes. Validated context commits independently of answer execution.
Task-aware preparation commits task selection and context as one checkpoint.
Every method opens a short transaction and closes its connection before return.
"""
import json
from dataclasses import asdict
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from app.context_proposal import RegisteredTarget, DialogueMessage, validate_context_value_contracts
from app.conversation_context import (
    ContextState, FactTarget, UserMessage, apply_context_updates, restore_context,
)
from app.analysis_preparation import resolve_analysis_request, guard_hypothetical_context_sources
from app.conversation_constraint_storage import ConversationConstraintStorage
from app.constraint_preparation import project_prepared_constraints
from app.task_constraints import restore_constraint_snapshot
from app.conversation_question_storage import ConversationQuestionStorage
from app.question_authority import restore_question_anchor


def _json_snapshot(value):
    """Use the same representation before and after the JSONB round trip."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _registry_records(registry):
    if (not isinstance(registry, tuple)
            or any(not isinstance(item, RegisteredTarget) or not isinstance(item.target, FactTarget)
                   or not isinstance(item.description, str) or not item.description.strip() for item in registry)):
        raise ValueError('invalid_context_registry')
    if len({item.target for item in registry}) != len(registry):
        raise ValueError('duplicate_registered_target')
    return _json_snapshot([{k: v for k, v in asdict(item).items()
                            if k != 'value_contract' or v is not None} for item in registry])


class PostgresConversationStore(ConversationConstraintStorage, ConversationQuestionStorage):
    def __init__(self, connect, *, schema: str):
        if (not isinstance(schema, str) or not schema.startswith('rag_conversation_')
                or not schema.replace('_', '').isalnum()):
            raise ValueError('require_dedicated_conversation_schema')
        self.connect, self.schema = connect, schema

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    def initialize(self):
        with self.connect() as conn:
            conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                input_version BIGINT NOT NULL DEFAULT 0 CHECK (input_version >= 0),
                current_run_id TEXT
            )''').format(self._table('conversations')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL REFERENCES {}(conversation_id),
                message_id TEXT NOT NULL, input_version BIGINT NOT NULL CHECK (input_version > 0),
                content TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (conversation_id, message_id), UNIQUE (conversation_id, input_version)
            )''').format(self._table('messages'), self._table('conversations')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL,
                input_version BIGINT NOT NULL CHECK (input_version > 0),
                base_version BIGINT NOT NULL CHECK (base_version >= 0 AND input_version = base_version + 1),
                format_version SMALLINT NOT NULL DEFAULT 1 CHECK (format_version = 1),
                proposal JSONB NOT NULL, snapshot JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (conversation_id, input_version),
                FOREIGN KEY (conversation_id, input_version) REFERENCES {}(conversation_id, input_version)
            )''').format(self._table('contexts'), self._table('messages')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL, task_id TEXT NOT NULL,
                goal TEXT NOT NULL CHECK (length(trim(goal)) > 0),
                source_message_id TEXT NOT NULL, source_quote TEXT NOT NULL,
                PRIMARY KEY (conversation_id, task_id),
                FOREIGN KEY (conversation_id, source_message_id)
                    REFERENCES {}(conversation_id, message_id)
            )''').format(self._table('tasks'), self._table('messages')))
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS registered_targets JSONB')
                .format(self._table('tasks')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL, input_version BIGINT NOT NULL,
                task_id TEXT, selection JSONB NOT NULL,
                PRIMARY KEY (conversation_id, input_version),
                FOREIGN KEY (conversation_id, input_version)
                    REFERENCES {}(conversation_id, input_version),
                FOREIGN KEY (conversation_id, task_id)
                    REFERENCES {}(conversation_id, task_id)
            )''').format(self._table('preparations'), self._table('contexts'), self._table('tasks')))
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS turn_question TEXT')
                .format(self._table('preparations')))
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS turn_request JSONB')
                .format(self._table('preparations')))
            for column in ('analysis_scope', 'analysis_request'):
                conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} JSONB')
                    .format(self._table('preparations'), sql.Identifier(column)))
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS question_anchor JSONB')
                .format(self._table('preparations')))
            self._initialize_constraints(conn)
            # Additive migration for the earlier explicit-task-only prototype.
            conn.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN task_id DROP NOT NULL')
                .format(self._table('preparations')))
            constraint = conn.execute('''SELECT 1 FROM pg_constraint c
                JOIN pg_class t ON t.oid=c.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace
                WHERE n.nspname=%s AND t.relname='preparations' AND c.conname='preparation_task_kind' ''',
                (self.schema,)).fetchone()
            if constraint is None:
                conn.execute(sql.SQL('''ALTER TABLE {} ADD CONSTRAINT preparation_task_kind CHECK (
                    ((task_id IS NULL AND selection->>'operation'='clarify') OR
                     (task_id IS NOT NULL AND selection->>'operation' IN ('create','select','resolve_task'))) IS TRUE
                )''').format(self._table('preparations')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL, clarification_version BIGINT NOT NULL,
                resolved_input_version BIGINT NOT NULL,
                PRIMARY KEY (conversation_id,clarification_version),
                UNIQUE (conversation_id,resolved_input_version),
                CHECK (resolved_input_version > clarification_version),
                FOREIGN KEY (conversation_id,clarification_version) REFERENCES {}(conversation_id,input_version),
                FOREIGN KEY (conversation_id,resolved_input_version) REFERENCES {}(conversation_id,input_version)
            )''').format(self._table('clarification_resolutions'), self._table('preparations'), self._table('preparations')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL REFERENCES {}(conversation_id),
                run_id TEXT NOT NULL, input_version BIGINT NOT NULL CHECK (input_version > 0),
                input_snapshot JSONB NOT NULL, build_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
                result JSONB, publication_at_finish TEXT,
                PRIMARY KEY (conversation_id, run_id), UNIQUE (conversation_id, input_version),
                FOREIGN KEY (conversation_id, input_version) REFERENCES {}(conversation_id, input_version),
                CHECK ((status = 'running' AND result IS NULL AND publication_at_finish IS NULL)
                    OR (status = 'completed' AND result IS NOT NULL
                        AND publication_at_finish IN ('current_at_commit', 'superseded'))
                    OR (status = 'failed' AND result IS NOT NULL AND publication_at_finish = 'failed'))
            )''').format(self._table('runs'), self._table('conversations'), self._table('messages')))
            conn.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS active_attempt_id TEXT')
                .format(self._table('runs')))
            conn.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
                conversation_id TEXT NOT NULL, run_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
                attempt_number BIGINT NOT NULL CHECK(attempt_number>0),
                status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running','completed','failed')),
                started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                lease_expires_at TIMESTAMPTZ NOT NULL, superseded_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ, result JSONB, decision TEXT,
                PRIMARY KEY(conversation_id,run_id,attempt_id),
                UNIQUE(conversation_id,run_id,attempt_number),
                FOREIGN KEY(conversation_id,run_id) REFERENCES {}(conversation_id,run_id),
                CHECK((status='running' AND result IS NULL AND finished_at IS NULL AND decision IS NULL)
                    OR (status IN ('completed','failed') AND result IS NOT NULL AND finished_at IS NOT NULL
                        AND decision IN ('accepted','lease_lost','run_finished')))
            )''').format(self._table('run_attempts'), self._table('runs')))

    def _conversation(self, conn, owner_id, conversation_id):
        row = conn.execute(sql.SQL('''SELECT input_version, current_run_id FROM {}
            WHERE conversation_id = %s AND owner_id = %s FOR UPDATE''')
            .format(self._table('conversations')), (conversation_id, owner_id)).fetchone()
        if row is None:
            raise ValueError('conversation_not_found')
        return row

    def create(self, owner_id: str) -> str:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError('require_trusted_owner_id')
        conversation_id = uuid4().hex
        with self.connect() as conn:
            conn.execute(sql.SQL('INSERT INTO {} (conversation_id, owner_id) VALUES (%s, %s)')
                .format(self._table('conversations')), (conversation_id, owner_id))
        return conversation_id

    def append_message(self, owner_id: str, conversation_id: str, message_id: str, content: str) -> dict:
        if any(not isinstance(v, str) or not v.strip() for v in (message_id, content)):
            raise ValueError('require_message_id_and_content')
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            old = conn.execute(sql.SQL('SELECT input_version, content FROM {} WHERE conversation_id=%s AND message_id=%s')
                .format(self._table('messages')), (conversation_id, message_id)).fetchone()
            if old is not None:
                if old[1] != content:
                    raise ValueError('message_id_conflict')
                return {'input_version': old[0], 'current_version': version, 'replayed': True}
            version += 1
            conn.execute(sql.SQL('INSERT INTO {} (conversation_id, message_id, input_version, content) VALUES (%s,%s,%s,%s)')
                .format(self._table('messages')), (conversation_id, message_id, version, content))
            # A new input invalidates the previous latest suggestion immediately.
            conn.execute(sql.SQL('UPDATE {} SET input_version=%s, current_run_id=NULL WHERE conversation_id=%s')
                .format(self._table('conversations')), (version, conversation_id))
        return {'input_version': version, 'current_version': version, 'replayed': False}

    def _latest_context(self, conn, conversation_id):
        row = conn.execute(sql.SQL('''SELECT input_version,snapshot FROM {}
            WHERE conversation_id=%s ORDER BY input_version DESC LIMIT 1''')
            .format(self._table('contexts')), (conversation_id,)).fetchone()
        if row is None:
            return ContextState()
        state = restore_context(row[1])
        if state.input_version != row[0]:
            raise ValueError('context_snapshot_version_mismatch')
        return state

    def _task(self, conn, conversation_id, task_id):
        row = conn.execute(sql.SQL('''SELECT task_id,goal,source_message_id,source_quote,registered_targets
            FROM {} WHERE conversation_id=%s AND task_id=%s''').format(self._table('tasks')),
            (conversation_id, task_id)).fetchone()
        if row is None:
            raise ValueError('task_not_found')
        task = dict(zip(('task_id', 'goal', 'source_message_id', 'source_quote'), row[:4]))
        if row[4] is not None:
            task['registered_targets'] = row[4]
        return task

    def _prepared_task(self, conn, conversation_id, input_version):
        row = conn.execute(sql.SQL('SELECT task_id FROM {} WHERE conversation_id=%s AND input_version=%s')
            .format(self._table('preparations')), (conversation_id, input_version)).fetchone()
        return self._task(conn, conversation_id, row[0]) if row and row[0] is not None else None

    def _preparation(self, conn, conversation_id, input_version):
        return conn.execute(sql.SQL('''SELECT task_id,selection,turn_question,turn_request,analysis_scope,analysis_request,
            evidence_request,incident_scope,constraint_snapshot,question_anchor FROM {}
            WHERE conversation_id=%s AND input_version=%s''').format(self._table('preparations')),
            (conversation_id, input_version)).fetchone()

    def _pending_task_clarifications(self, conn, conversation_id):
        rows = conn.execute(sql.SQL('''SELECT p.input_version,m.message_id,m.content,p.selection
            FROM {} p JOIN {} m ON m.conversation_id=p.conversation_id AND m.input_version=p.input_version
            LEFT JOIN {} r ON r.conversation_id=p.conversation_id AND r.clarification_version=p.input_version
            WHERE p.conversation_id=%s AND p.selection->>'operation'='clarify'
                AND r.resolved_input_version IS NULL ORDER BY p.input_version''')
            .format(self._table('preparations'), self._table('messages'), self._table('clarification_resolutions')),
            (conversation_id,)).fetchall()
        return [{'input_version': v, 'message_id': mid, 'content': content,
                 'candidate_task_ids': selection['candidate_task_ids'], 'question': selection['question']}
                for v, mid, content, selection in rows]

    def _has_preparations(self, conn, conversation_id):
        return conn.execute(sql.SQL('SELECT 1 FROM {} WHERE conversation_id=%s LIMIT 1')
            .format(self._table('preparations')), (conversation_id,)).fetchone() is not None

    def _task_catalog(self, conn, conversation_id, base_version):
        """Read each task's latest prepared question in the same authorized view."""
        rows = conn.execute(sql.SQL('''SELECT t.task_id,t.goal,t.source_message_id,t.source_quote,
                p.input_version,p.turn_question,t.registered_targets,m.input_version,p.analysis_scope,p.question_anchor FROM {} t
                JOIN {} m ON m.conversation_id=t.conversation_id AND m.message_id=t.source_message_id
                JOIN LATERAL (
                    SELECT input_version,turn_question,analysis_scope,question_anchor FROM {} p
                    WHERE p.conversation_id=t.conversation_id AND p.task_id=t.task_id
                      AND p.input_version<=%s ORDER BY p.input_version DESC LIMIT 1
                ) p ON TRUE WHERE t.conversation_id=%s ORDER BY m.input_version,t.task_id''')
            .format(self._table('tasks'), self._table('messages'), self._table('preparations')),
            (base_version, conversation_id)).fetchall()
        records = [dict(zip(('task_id', 'goal', 'source_message_id', 'source_quote',
                          'latest_input_version', 'latest_question', 'registered_targets', 'created_input_version',
                          'latest_analysis_scope','latest_question_anchor'), row)) for row in rows]
        for record in records:
            if record['latest_question_anchor'] is not None:
                anchor=restore_question_anchor(record['latest_question_anchor'],task_id=record['task_id'],
                                               max_input_version=record['latest_input_version'])
                record['latest_retrieval_question']=record['latest_question']
                record['latest_question']=anchor.text
        return records

    def _analysis_catalog(self, conn, conversation_id, base_version, task_id=None):
        rows = conn.execute(sql.SQL('''SELECT task_id,input_version,turn_question,analysis_scope FROM {}
            WHERE conversation_id=%s AND input_version<=%s AND analysis_scope IS NOT NULL
              AND (%s::text IS NULL OR task_id=%s)
            ORDER BY input_version''').format(self._table('preparations')),
            (conversation_id, base_version, task_id, task_id)).fetchall()
        return [dict(zip(('task_id','input_version','question','analysis_scope'), r)) for r in rows]

    def read_preparation(self, owner_id: str, conversation_id: str) -> dict:
        """Return the last committed context and the next unprocessed user message.

        No connection/lock is held while the caller asks a model for a proposal.
        Accepted input may be ahead of prepared context; never skip that gap.
        """
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            state = self._latest_context(conn, conversation_id)
            row = conn.execute(sql.SQL('''SELECT message_id,input_version,content FROM {}
                WHERE conversation_id=%s AND input_version=%s''').format(self._table('messages')),
                (conversation_id, state.input_version + 1)).fetchone()
            message = UserMessage(*row) if row is not None else None
            task = self._prepared_task(conn, conversation_id, state.input_version)
            prepared = self._preparation(conn, conversation_id, state.input_version)
            pending = self._pending_task_clarifications(conn, conversation_id)
            task_catalog = self._task_catalog(conn, conversation_id, state.input_version)
            analysis_catalog = self._analysis_catalog(conn, conversation_id, state.input_version)
            constraint_view = self._constraint_view(conn,conversation_id,state.input_version)
        return {'current_version': version, 'state': state, 'next_message': message,
                'prepared_task': task,
                'task_catalog': task_catalog,
                'turn_question': prepared[2] if prepared else None,
                'question_anchor': prepared[9] if prepared else None,
                'turn_request': prepared[3] if prepared else None,
                'analysis_scope': prepared[4] if prepared else None,
                'analysis_catalog': analysis_catalog, **constraint_view,
                'incident_scope': prepared[7] if prepared else None,
                'ready_for_run': (task is not None and bool(prepared[2]) and state.input_version == version
                                  and (prepared[4] or {}).get('mode') != 'needs_clarification'
                                  and (prepared[7] or {}).get('mode') != 'needs_clarification'),
                'prepared_status': ('needs_task' if prepared and prepared[0] is None else
                                    'needs_question' if prepared and not prepared[2] else
                                    'needs_analysis_scope' if prepared and (prepared[4] or {}).get('mode') == 'needs_clarification' else
                                    'needs_incident' if prepared and (prepared[7] or {}).get('mode') == 'needs_clarification' else
                                    'ready' if prepared else 'not_prepared'),
                'pending_task_clarifications': pending}

    def read_dialogue(self, owner_id, conversation_id, base_version):
        """Canonical prior user messages and actually published assistant replies."""
        with self.connect() as conn:
            self._conversation(conn, owner_id, conversation_id)
            rows = conn.execute(sql.SQL('''SELECT m.message_id,m.content,p.selection,r.result,p.analysis_scope,p.incident_scope
                FROM {} m LEFT JOIN {} p USING(conversation_id,input_version)
                LEFT JOIN {} r ON r.conversation_id=m.conversation_id AND r.input_version=m.input_version
                    AND r.status='completed' AND r.publication_at_finish='current_at_commit'
                WHERE m.conversation_id=%s AND m.input_version<=%s ORDER BY m.input_version''')
                .format(self._table('messages'), self._table('preparations'), self._table('runs')),
                (conversation_id, base_version)).fetchall()
        dialogue, used = [], {row[0] for row in rows}
        for mid, content, selection, result, scope, incident_scope in rows:
            dialogue.append(DialogueMessage(mid, 'user', content))
            reply = (selection['question'] if selection and selection['operation'] == 'clarify'
                     else scope['question'] if scope and scope.get('mode') == 'needs_clarification'
                     else incident_scope['question'] if incident_scope and incident_scope.get('mode') == 'needs_clarification'
                     else (result or {}).get('message'))
            if reply:
                assistant_id = 'assistant_for_' + mid
                while assistant_id in used:
                    assistant_id += '_'
                used.add(assistant_id)
                dialogue.append(DialogueMessage(assistant_id, 'assistant', reply))
        return tuple(dialogue)

    def read_message_turn(self, owner_id, conversation_id, message_id):
        """Read a receipt/result by canonical message ID, checking ownership first."""
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            row = conn.execute(sql.SQL('''SELECT input_version,content FROM {}
                WHERE conversation_id=%s AND message_id=%s''').format(self._table('messages')),
                (conversation_id, message_id)).fetchone()
            if row is None:
                raise ValueError('message_not_found')
            prepared = self._preparation(conn, conversation_id, row[0])
            run = conn.execute(sql.SQL('''SELECT run_id,input_version,build_id,status,result,publication_at_finish
                FROM {} WHERE conversation_id=%s AND input_version=%s''').format(self._table('runs')),
                (conversation_id, row[0])).fetchone()
        return {'conversation_id': conversation_id, 'message_id': message_id,
            'input_version': row[0], 'content': row[1], 'current_version': version,
            'selection': prepared[1] if prepared else None, 'turn_question': prepared[2] if prepared else None,
            'analysis_scope': prepared[4] if prepared else None,
            'analysis_request': prepared[5] if prepared else None,
            'question_anchor': prepared[9] if prepared else None,
            'evidence_request': prepared[6] if prepared else None,
            'incident_scope': prepared[7] if prepared else None,
            'run': dict(zip(('run_id','input_version','build_id','status','result','publication_at_finish'), run)) if run else None}

    def commit_context_update(self, owner_id: str, conversation_id: str, message_id: str,
                              base_version: int, proposal: dict, *,
                              allowed_targets: frozenset[FactTarget]) -> dict:
        """Validate a patch against canonical input, then durably checkpoint it.

        This short transaction contains only reads, pure validation and a write.
        Message receipt, context preparation and RAG completion are separate.
        Always use the returned canonical state: IDs are allocated at commit,
        and replay returns the saved state instead of allocating them again.
        """
        if type(base_version) is not int or base_version < 0:
            raise ValueError('invalid_context_base_version')
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            if self._has_preparations(conn, conversation_id):
                raise ValueError('use_atomic_task_preparation')
            message = self._message(conn, conversation_id, message_id, base_version)
            state, replayed = self._commit_context(conn, conversation_id, message, base_version, proposal, allowed_targets)
        return {'state': state, 'current_version': version, 'replayed': replayed,
                'is_latest_input': state.input_version == version}

    def _message(self, conn, conversation_id, message_id, base_version):
        row = conn.execute(sql.SQL('''SELECT message_id,input_version,content FROM {}
            WHERE conversation_id=%s AND message_id=%s''').format(self._table('messages')),
            (conversation_id, message_id)).fetchone()
        if row is None:
            raise ValueError('message_not_found')
        message = UserMessage(*row)
        if message.input_version != base_version + 1:
            raise ValueError('message_version_not_next')
        return message

    def _commit_context(self, conn, conversation_id, message, base_version, proposal, allowed_targets,
                        *, deferred_message=None):
        """Called only under the authorized conversation lock; never commits itself."""
        saved = conn.execute(sql.SQL('''SELECT proposal,snapshot FROM {}
            WHERE conversation_id=%s AND input_version=%s''').format(self._table('contexts')),
            (conversation_id, message.input_version)).fetchone()
        if saved is not None:
            if saved[0] != proposal:
                raise ValueError('context_update_conflict')
            return restore_context(saved[1]), True
        previous = self._latest_context(conn, conversation_id)
        if previous.input_version != base_version:
            raise ValueError('context_version_conflict')
        if deferred_message is not None:
            self._guard_deferred_updates(conn, conversation_id, previous, proposal, deferred_message)
        state = apply_context_updates(previous, message, proposal, allowed_targets=allowed_targets,
                                      deferred_message=deferred_message)
        conn.execute(sql.SQL('''INSERT INTO {} (conversation_id,input_version,base_version,proposal,snapshot)
            VALUES (%s,%s,%s,%s,%s)''').format(self._table('contexts')),
            (conversation_id, state.input_version, base_version, Jsonb(proposal), Jsonb(state.to_dict())))
        return state, False

    def _guard_deferred_updates(self, conn, conversation_id, previous, proposal, deferred_message):
        """An old supplement must not silently replace a newer statement/dispute."""
        updates = proposal.get('updates', []) if isinstance(proposal, dict) else []
        if not isinstance(updates, list):
            return  # The normal context validator reports malformed proposals.
        later = conn.execute(sql.SQL('''SELECT proposal FROM {} WHERE conversation_id=%s
            AND input_version>%s AND input_version<=%s''').format(self._table('contexts')),
            (conversation_id, deferred_message.input_version, previous.input_version)).fetchall()
        for update in updates:
            if (not isinstance(update, dict) or not isinstance(update.get('source'), dict)
                    or update['source'].get('message_id') != deferred_message.message_id):
                continue
            for item in (*previous.current_facts, *previous.pending_facts):
                if update.get('target') != asdict(item.target):
                    continue
                row = conn.execute(sql.SQL('SELECT input_version FROM {} WHERE conversation_id=%s AND message_id=%s')
                    .format(self._table('messages')), (conversation_id, item.source.message_id)).fetchone()
                if row is None:
                    raise ValueError('stored_fact_source_not_found')
                if row[0] > deferred_message.input_version:
                    raise ValueError('deferred_source_stale_for_target')
            # Same-value confirmations deliberately retain their fact ID/source.
            # Their proposals are still newer statements and must also protect
            # the target against an older supplement that is only now attributed.
            for (later_proposal,) in later:
                for newer in later_proposal['updates']:
                    if newer['target'] != update.get('target'):
                        continue
                    row = conn.execute(sql.SQL('SELECT input_version FROM {} WHERE conversation_id=%s AND message_id=%s')
                        .format(self._table('messages')), (conversation_id, newer['source']['message_id'])).fetchone()
                    if row is None:
                        raise ValueError('stored_fact_source_not_found')
                    if row[0] > deferred_message.input_version:
                        raise ValueError('deferred_source_stale_for_target')

    def _resolution_source(self, conn, conversation_id, message, selection):
        source_version = selection['clarification_version']
        pending = self._preparation(conn, conversation_id, source_version)
        if (source_version >= message.input_version or pending is None
                or pending[1]['operation'] != 'clarify'):
            raise ValueError('task_clarification_not_found')
        if selection['task_id'] not in pending[1]['candidate_task_ids']:
            raise ValueError('task_not_in_clarification_candidates')
        resolved = conn.execute(sql.SQL('SELECT 1 FROM {} WHERE conversation_id=%s AND clarification_version=%s')
            .format(self._table('clarification_resolutions')), (conversation_id, source_version)).fetchone()
        if resolved:
            raise ValueError('task_clarification_already_resolved')
        row = conn.execute(sql.SQL('''SELECT message_id,input_version,content FROM {}
            WHERE conversation_id=%s AND input_version=%s''').format(self._table('messages')),
            (conversation_id, source_version)).fetchone()
        return UserMessage(*row)

    def _resolve_turn_question(self, conn, conversation_id, message, task_id, request, *, deferred_message=None):
        """Resolve an explicit new focus or same-task continuation; never guess intent."""
        if task_id is None:
            if request is not None:
                raise ValueError('ambiguous_task_has_no_turn_question')
            return None
        if not isinstance(request, dict):
            raise ValueError('require_turn_request')
        mode = request.get('mode')
        if mode == 'question':
            if (set(request) != {'mode', 'question', 'source'}
                    or not isinstance(request['question'], str) or not request['question'].strip()):
                raise ValueError('invalid_turn_question')
            source = request['source']
            authorized = {message.message_id: message}
            if deferred_message is not None:
                authorized[deferred_message.message_id] = deferred_message
            if (not isinstance(source, dict) or set(source) != {'message_id', 'quote'}
                    or not isinstance(source['message_id'], str) or source['message_id'] not in authorized
                    or not isinstance(source['quote'], str) or not source['quote'].strip()
                    or source['quote'] not in authorized[source['message_id']].content):
                raise ValueError('turn_question_source_not_authorized')
            return request['question']
        if mode == 'continue':
            if (set(request) != {'mode', 'from_input_version'}
                    or type(request['from_input_version']) is not int
                    or not 0 < request['from_input_version'] < message.input_version):
                raise ValueError('invalid_question_continuation')
            previous = conn.execute(sql.SQL('''SELECT input_version,turn_question FROM {}
                WHERE conversation_id=%s AND task_id=%s AND input_version<%s
                ORDER BY input_version DESC LIMIT 1''').format(self._table('preparations')),
                (conversation_id, task_id, message.input_version)).fetchone()
            if previous is None or previous[0] != request['from_input_version']:
                raise ValueError('question_continuation_must_use_latest_same_task')
            if not previous[1]:
                raise ValueError('previous_turn_question_missing')
            return previous[1]
        raise ValueError('invalid_turn_request_mode')

    def commit_preparation(self, owner_id: str, conversation_id: str, message_id: str,
                           base_version: int, proposal: dict, selection: dict, *,
                           allowed_targets: frozenset[FactTarget], turn_request: dict | None = None,
                           registry: tuple[RegisteredTarget, ...] | None = None,
                           analysis_request: dict | None = None,
                           evidence_request: dict | None = None) -> dict:
        """Atomically select/create a task and save the message's updated context.

        Selection is an explicit trusted-caller proposal, not automatic intent
        recognition. Quotes validate provenance, not the semantics of the choice.
        No separate active-task pointer: the committed preparation is authoritative.
        Legacy context-only conversations require an explicit migration, not a
        guessed assignment of historical checkpoints to a task.
        Each selected task also requires an explicit per-turn question or a
        same-task continuation reference. The long-term goal stays unchanged.
        """
        if type(base_version) is not int or base_version < 0:
            raise ValueError('invalid_context_base_version')
        registration = _registry_records(registry) if registry is not None else None
        if registry is not None and frozenset(item.target for item in registry) != allowed_targets:
            raise ValueError('task_registry_and_allowed_targets_differ')
        if not isinstance(selection, dict):
            raise ValueError('invalid_task_selection')
        operation = selection.get('operation')
        fields = {
            'create': {'operation', 'source_quote', 'goal'},
            'select': {'operation', 'source_quote', 'task_id'},
            'clarify': {'operation', 'source_quote', 'candidate_task_ids', 'question'},
            'resolve_task': {'operation', 'source_quote', 'task_id', 'clarification_version'},
        }
        if not isinstance(operation, str) or operation not in fields or set(selection) != fields[operation]:
            raise ValueError('invalid_task_selection')
        text_fields = fields[operation] - {'candidate_task_ids', 'clarification_version'}
        if any(not isinstance(selection[k], str) or not selection[k].strip() for k in text_fields):
            raise ValueError('invalid_task_selection')
        if operation == 'clarify':
            candidates = selection['candidate_task_ids']
            if (not isinstance(candidates, list) or len(candidates) < 2
                    or any(not isinstance(c, str) or not c.strip() for c in candidates)
                    or len(set(candidates)) != len(candidates)):
                raise ValueError('invalid_task_candidates')
            if proposal != {'updates': []}:
                raise ValueError('ambiguous_task_cannot_update_context')
            if analysis_request is not None:
                raise ValueError('ambiguous_task_has_no_analysis_scope')
            if evidence_request is not None:
                raise ValueError('ambiguous_task_has_no_incident_scope')
        if operation == 'resolve_task' and (type(selection['clarification_version']) is not int
                                           or selection['clarification_version'] < 1):
            raise ValueError('invalid_clarification_version')
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            message = self._message(conn, conversation_id, message_id, base_version)
            if selection['source_quote'] not in message.content:
                raise ValueError('task_quote_not_in_current_user_message')
            saved = self._preparation(conn, conversation_id, message.input_version)
            if saved:
                if saved[1] != selection:
                    raise ValueError('task_selection_conflict')
                if saved[3] != turn_request:
                    raise ValueError('turn_question_conflict')
                if saved[5] != analysis_request:
                    raise ValueError('analysis_scope_conflict')
                if saved[6] != evidence_request:
                    raise ValueError('evidence_request_conflict')
                state, _ = self._commit_context(conn, conversation_id, message, base_version, proposal, allowed_targets)
                task = self._task(conn, conversation_id, saved[0]) if saved[0] is not None else None
                if registration is not None and task is not None and task.get('registered_targets') != registration:
                    raise ValueError('task_registry_conflict')
                turn_question = saved[2]
                analysis_scope = saved[4]
                incident_scope, constraint_snapshot = saved[7],saved[8]
                question_anchor = saved[9]
            else:
                if base_version and self._preparation(conn, conversation_id, base_version) is None:
                    raise ValueError('previous_turn_not_prepared')
                deferred_message = None
                if operation == 'create':
                    task_id = uuid4().hex
                    conn.execute(sql.SQL('''INSERT INTO {}
                        (conversation_id,task_id,goal,source_message_id,source_quote) VALUES (%s,%s,%s,%s,%s)''')
                        .format(self._table('tasks')),
                        (conversation_id, task_id, selection['goal'], message_id, selection['source_quote']))
                    task = self._task(conn, conversation_id, task_id)
                elif operation == 'clarify':
                    for candidate in selection['candidate_task_ids']:
                        self._task(conn, conversation_id, candidate)
                    task_id, task = None, None
                else:
                    task_id = selection['task_id']
                    task = self._task(conn, conversation_id, task_id)
                    if operation == 'resolve_task':
                        deferred_message = self._resolution_source(conn, conversation_id, message, selection)
                if task is not None:
                    existing = task.get('registered_targets')
                    if existing is not None:
                        canonical_targets = frozenset(FactTarget(**r['target']) for r in existing)
                        if allowed_targets != canonical_targets or (registration is not None and registration != existing):
                            raise ValueError('task_registry_conflict')
                    elif registration is not None:
                        conn.execute(sql.SQL('UPDATE {} SET registered_targets=%s WHERE conversation_id=%s AND task_id=%s')
                            .format(self._table('tasks')), (Jsonb(registration), conversation_id, task_id))
                        task = self._task(conn, conversation_id, task_id)
                turn_question = self._resolve_turn_question(conn, conversation_id, message, task_id, turn_request,
                                                            deferred_message=deferred_message)
                question_anchor=self._question_anchor(conn,conversation_id,task_id,message,turn_request,deferred_message)
                # Resolve again from canonical same-task rows under the same
                # conversation lock as the context/preparation commit.
                analysis_scope = None
                if analysis_request is not None:
                    if task is None or task.get('registered_targets') is None:
                        raise ValueError('require_analysis_target_registry')
                    analysis_scope = resolve_analysis_request(analysis_request, message,
                        allowed_targets=allowed_targets,
                        history=self._analysis_catalog(conn, conversation_id, base_version, task_id),
                        deferred_message=deferred_message)
                    guard_hypothetical_context_sources(proposal, analysis_scope)
                incident_scope, constraint_snapshot = None,None
                if evidence_request is not None:
                    if task is None:
                        raise ValueError('require_selected_task_constraint_state')
                    incident_scope, constraint_snapshot = self._commit_evidence_request(conn,
                        conversation_id,task_id,message,base_version,evidence_request,deferred_message)
                state, replayed_context = self._commit_context(
                    conn, conversation_id, message, base_version, proposal, allowed_targets,
                    deferred_message=deferred_message)
                if task is not None and task.get('registered_targets') is not None:
                    canonical_registry = tuple(RegisteredTarget(FactTarget(**r['target']),r['description'],
                        r.get('value_contract')) for r in task['registered_targets'])
                    # Recheck after source/shape validation inside the same short
                    # transaction. A bypass of model preparation cannot commit
                    # invalid values; all task/incident/context writes roll back.
                    validate_context_value_contracts(proposal,canonical_registry,
                        analysis_request=analysis_request,evidence_request=evidence_request)
                if replayed_context:
                    raise ValueError('context_already_committed_without_task')
                conn.execute(sql.SQL('''INSERT INTO {}
                    (conversation_id,input_version,task_id,selection,turn_question,turn_request,analysis_scope,analysis_request,
                     evidence_request,incident_scope,constraint_snapshot,question_anchor)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''').format(self._table('preparations')),
                    (conversation_id, state.input_version, task_id, Jsonb(selection), turn_question,
                     Jsonb(turn_request) if turn_request is not None else None,
                     Jsonb(analysis_scope) if analysis_scope is not None else None,
                     Jsonb(analysis_request) if analysis_request is not None else None,
                     Jsonb(evidence_request) if evidence_request is not None else None,
                     Jsonb(incident_scope) if incident_scope is not None else None,
                     Jsonb(constraint_snapshot) if constraint_snapshot is not None else None,
                     Jsonb(question_anchor) if question_anchor is not None else None))
                if operation == 'resolve_task':
                    conn.execute(sql.SQL('''INSERT INTO {}
                        (conversation_id,clarification_version,resolved_input_version) VALUES (%s,%s,%s)''')
                        .format(self._table('clarification_resolutions')),
                        (conversation_id, selection['clarification_version'], state.input_version))
        return {'task': task, 'state': state, 'current_version': version,
                'turn_question': turn_question, 'turn_request': _json_snapshot(turn_request),
                'analysis_scope': _json_snapshot(analysis_scope), 'analysis_request': _json_snapshot(analysis_request),
                'incident_scope': _json_snapshot(incident_scope),
                'constraint_snapshot': _json_snapshot(constraint_snapshot),
                'question_anchor': _json_snapshot(question_anchor),
                'replayed': saved is not None,
                'ready_for_run': (task is not None and bool(turn_question) and state.input_version == version
                                  and (analysis_scope or {}).get('mode') != 'needs_clarification'
                                  and (incident_scope or {}).get('mode') != 'needs_clarification'),
                'status': ('needs_task' if task is None else 'needs_analysis_scope'
                           if (analysis_scope or {}).get('mode') == 'needs_clarification'
                           else 'needs_incident' if (incident_scope or {}).get('mode') == 'needs_clarification'
                           else 'ready' if turn_question else 'needs_question')}

    def start_prepared_run(self, owner_id: str, conversation_id: str,
                           input_version: int, build_id: str) -> str:
        return self._start_prepared(owner_id, conversation_id, input_version, build_id, claim=False)

    def claim_prepared_run(self, owner_id, conversation_id, input_version, build_id):
        """Exactly one caller creates this execution; duplicates only read its state."""
        return self._start_prepared(owner_id, conversation_id, input_version, build_id, claim=True)

    def _start_prepared(self, owner_id, conversation_id, input_version, build_id, *, claim):
        """Freeze canonical task/context only after full preparation has committed."""
        if type(input_version) is not int or input_version < 1 or not isinstance(build_id, str) or not build_id.strip():
            raise ValueError('invalid_run_input')
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            if version != input_version:
                raise ValueError('conversation_changed')
            task = self._prepared_task(conn, conversation_id, input_version)
            state = self._latest_context(conn, conversation_id)
            if task is None or state.input_version != input_version:
                raise ValueError('turn_not_prepared')
            prepared = self._preparation(conn, conversation_id, input_version)
            if not prepared[2]:
                raise ValueError('turn_question_not_prepared')
            if (prepared[4] or {}).get('mode') == 'needs_clarification':
                raise ValueError('analysis_scope_not_ready')
            if (prepared[7] or {}).get('mode') == 'needs_clarification':
                raise ValueError('incident_scope_not_ready')
            raw = conn.execute(sql.SQL('SELECT message_id,content FROM {} WHERE conversation_id=%s AND input_version=%s')
                .format(self._table('messages')), (conversation_id, input_version)).fetchone()
            if 'registered_targets' in task:
                targets = frozenset(FactTarget(**r['target']) for r in task['registered_targets'])
                state = ContextState(state.input_version, tuple(f for f in state.current_facts if f.target in targets), (),
                                     tuple(p for p in state.pending_facts if p.target in targets))
            snapshot = _json_snapshot({'task': task, 'context': state.to_dict(), 'turn': {
                'input_version': input_version, 'message_id': raw[0], 'raw_message': raw[1],
                'question': prepared[2], 'request': prepared[3]}})
            if prepared[9] is not None:
                anchor=restore_question_anchor(prepared[9],task_id=task['task_id'],max_input_version=input_version)
                snapshot['turn']['question_anchor']=anchor.to_dict()
                snapshot['turn']['retrieval_question']=prepared[2]
                snapshot['turn']['question']=anchor.text
            if prepared[4] is not None:
                snapshot['turn']['analysis_scope'] = _json_snapshot(prepared[4])
                snapshot['turn']['analysis_request'] = _json_snapshot(prepared[5])
            if prepared[6] is not None:
                constraints = restore_constraint_snapshot(prepared[8])
                if constraints.input_version != input_version:
                    raise ValueError('constraint_snapshot_version_mismatch')
                snapshot['turn']['task_constraints'] = _json_snapshot(
                    project_prepared_constraints(constraints,prepared[7],task['task_id']))
                snapshot['turn']['evidence_request'] = _json_snapshot(prepared[6])
            return self._start_run(conn, conversation_id, input_version, snapshot, build_id, claim=claim)

    def read_run_input(self, owner_id: str, conversation_id: str, run_id: str) -> dict:
        """Read the authorized immutable execution input; no model call under lock."""
        with self.connect() as conn:
            self._conversation(conn, owner_id, conversation_id)
            row = conn.execute(sql.SQL('''SELECT input_version,input_snapshot,build_id,status,result,publication_at_finish FROM {}
                WHERE conversation_id=%s AND run_id=%s''').format(self._table('runs')),
                (conversation_id, run_id)).fetchone()
        if row is None:
            raise ValueError('run_not_found')
        return {'run_id': run_id, **dict(zip(('input_version', 'input_snapshot', 'build_id', 'status',
                                             'result', 'publication_at_finish'), row))}

    def start_run(self, owner_id: str, conversation_id: str, input_version: int,
                  input_snapshot: dict, build_id: str) -> str:
        if (type(input_version) is not int or input_version < 1 or not isinstance(input_snapshot, dict)
                or not isinstance(build_id, str) or not build_id.strip()):
            raise ValueError('invalid_run_input')
        input_snapshot = _json_snapshot(input_snapshot)
        with self.connect() as conn:
            version, _ = self._conversation(conn, owner_id, conversation_id)
            if version != input_version:
                raise ValueError('conversation_changed')
            if self._has_preparations(conn, conversation_id):
                raise ValueError('use_prepared_run_entry')
            return self._start_run(conn, conversation_id, input_version, input_snapshot, build_id)

    def _start_run(self, conn, conversation_id, input_version, input_snapshot, build_id, *, claim=False):
        old = conn.execute(sql.SQL('''SELECT run_id, input_snapshot, build_id FROM {}
            WHERE conversation_id=%s AND input_version=%s''').format(self._table('runs')),
            (conversation_id, input_version)).fetchone()
        if old is not None:
            if old[1:] != (input_snapshot, build_id):
                raise ValueError('run_input_conflict')
            return {'run_id': old[0], 'created': False} if claim else old[0]
        run_id = uuid4().hex
        conn.execute(sql.SQL('''INSERT INTO {} (conversation_id,run_id,input_version,input_snapshot,build_id)
            VALUES (%s,%s,%s,%s,%s)''').format(self._table('runs')),
            (conversation_id, run_id, input_version, Jsonb(input_snapshot), build_id))
        return {'run_id': run_id, 'created': True} if claim else run_id

    def finish_run(self, owner_id: str, conversation_id: str, run_id: str,
                   result: dict, *, status: str = 'completed') -> dict:
        if status not in ('completed', 'failed') or not isinstance(result, dict):
            raise ValueError('invalid_run_result')
        with self.connect() as conn:
            version, current = self._conversation(conn, owner_id, conversation_id)
            token = conn.execute(sql.SQL('SELECT active_attempt_id FROM {} WHERE conversation_id=%s AND run_id=%s')
                .format(self._table('runs')), (conversation_id, run_id)).fetchone()
            if token and token[0] is not None:
                raise ValueError('attempt_token_required')
            return self._finish_run(conn, conversation_id, run_id, result, status, version, current)

    def _finish_run(self, conn, conversation_id, run_id, result, status, version, current):
            row = conn.execute(sql.SQL('''SELECT input_version,status,result,publication_at_finish FROM {}
                WHERE conversation_id=%s AND run_id=%s''').format(self._table('runs')),
                (conversation_id, run_id)).fetchone()
            if row is None:
                raise ValueError('run_not_found')
            run_version, old_status, old_result, publication = row
            replayed = old_status != 'running'
            if replayed:
                if old_status != status or old_result != result:
                    raise ValueError('run_result_conflict')
            else:
                publication = ('failed' if status == 'failed' else
                               'current_at_commit' if version == run_version else 'superseded')
                conn.execute(sql.SQL('''UPDATE {} SET status=%s,result=%s,publication_at_finish=%s
                    WHERE conversation_id=%s AND run_id=%s''').format(self._table('runs')),
                    (status, Jsonb(result), publication, conversation_id, run_id))
                if publication == 'current_at_commit':
                    conn.execute(sql.SQL('UPDATE {} SET current_run_id=%s WHERE conversation_id=%s AND input_version=%s')
                        .format(self._table('conversations')), (run_id, conversation_id, run_version))
                    current = run_id
            return {'run_id': run_id, 'input_version': run_version, 'current_version': version,
                    'status': status, 'publication_at_finish': publication,
                    'is_current': current == run_id and version == run_version and status == 'completed',
                    'replayed': replayed}

    def read_current(self, owner_id: str, conversation_id: str) -> dict:
        # One statement reads a consistent snapshot; old persisted flags are not enough.
        with self.connect() as conn:
            row = conn.execute(sql.SQL('''SELECT c.input_version,r.run_id,r.result FROM {} c
                LEFT JOIN {} r ON r.conversation_id=c.conversation_id AND r.run_id=c.current_run_id
                    AND r.input_version=c.input_version AND r.status='completed'
                WHERE c.conversation_id=%s AND c.owner_id=%s''')
                .format(self._table('conversations'), self._table('runs')),
                (conversation_id, owner_id)).fetchone()
        if row is None:
            raise ValueError('conversation_not_found')
        return {'input_version': row[0], 'run_id': row[1], 'result': row[2]}

    def read_history(self, owner_id: str, conversation_id: str) -> dict:
        """Internal audit view; not a public API response or a model context builder."""
        with self.connect() as conn:
            self._conversation(conn, owner_id, conversation_id)
            messages = conn.execute(sql.SQL('''SELECT message_id,input_version,content FROM {}
                WHERE conversation_id=%s ORDER BY input_version''').format(self._table('messages')),
                (conversation_id,)).fetchall()
            runs = conn.execute(sql.SQL('''SELECT run_id,input_version,input_snapshot,build_id,status,result,publication_at_finish
                FROM {} WHERE conversation_id=%s ORDER BY input_version''').format(self._table('runs')),
                (conversation_id,)).fetchall()
            contexts = conn.execute(sql.SQL('''SELECT input_version,base_version,proposal,snapshot FROM {}
                WHERE conversation_id=%s ORDER BY input_version''').format(self._table('contexts')),
                (conversation_id,)).fetchall()
            tasks = conn.execute(sql.SQL('''SELECT task_id,goal,source_message_id,source_quote FROM {}
                WHERE conversation_id=%s ORDER BY task_id''').format(self._table('tasks')),
                (conversation_id,)).fetchall()
            preparations = conn.execute(sql.SQL('''SELECT input_version,task_id,selection,turn_question,turn_request,analysis_scope,analysis_request,
                evidence_request,incident_scope,constraint_snapshot,question_anchor FROM {}
                WHERE conversation_id=%s ORDER BY input_version''').format(self._table('preparations')),
                (conversation_id,)).fetchall()
            resolutions = conn.execute(sql.SQL('''SELECT clarification_version,resolved_input_version FROM {}
                WHERE conversation_id=%s ORDER BY resolved_input_version''')
                .format(self._table('clarification_resolutions')), (conversation_id,)).fetchall()
            attempts = conn.execute(sql.SQL('''SELECT run_id,attempt_id,attempt_number,status,started_at,
                lease_expires_at,superseded_at,finished_at,result,decision FROM {}
                WHERE conversation_id=%s ORDER BY run_id,attempt_number''').format(self._table('run_attempts')),
                (conversation_id,)).fetchall()
        return {'messages': [dict(zip(('message_id','input_version','content'), r)) for r in messages],
                'contexts': [dict(zip(('input_version','base_version','proposal','snapshot'), r)) for r in contexts],
                'tasks': [dict(zip(('task_id','goal','source_message_id','source_quote'), r)) for r in tasks],
                'preparations': [dict(zip(('input_version','task_id','selection','turn_question','turn_request',
                                         'analysis_scope','analysis_request','evidence_request',
                                         'incident_scope','constraint_snapshot','question_anchor'), r)) for r in preparations],
                'task_resolutions': [dict(zip(('clarification_version','resolved_input_version'), r)) for r in resolutions],
                'attempts': [dict(zip(('run_id','attempt_id','attempt_number','status','started_at','lease_expires_at',
                    'superseded_at','finished_at','result','decision'),
                    (v.isoformat() if hasattr(v, 'isoformat') else v for v in row))) for row in attempts],
                'runs': [dict(zip(('run_id','input_version','input_snapshot','build_id','status','result','publication_at_finish'), r))
                         for r in runs]}
