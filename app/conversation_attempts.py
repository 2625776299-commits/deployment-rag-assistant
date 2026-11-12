"""Short SQL transactions for leased attempts and fenced publication of one run.

Database time decides lease validity. A new attempt never changes the logical
run's frozen input/build. Stale results remain audit history, not current output.
"""
import math
from threading import Event, Thread
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb


def lease_duration(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid_attempt_lease')
    return seconds


class RunAttempts:
    def __init__(self, store):
        self.store = store

    def claim(self, owner_id, conversation_id, run_id, *, lease_seconds=90):
        lease_duration(lease_seconds)
        s = self.store
        with s.connect() as conn:
            version, _ = s._conversation(conn, owner_id, conversation_id)
            run = conn.execute(sql.SQL('SELECT input_version,status,active_attempt_id FROM {} WHERE conversation_id=%s AND run_id=%s')
                .format(s._table('runs')), (conversation_id, run_id)).fetchone()
            if run is None:
                raise ValueError('run_not_found')
            if run[1] != 'running' or run[0] != version:
                return {'acquired': False, 'reason': 'run_finished' if run[1] != 'running' else 'superseded_message'}
            # Read the clock AFTER acquiring the shared conversation lock.
            now = conn.execute('SELECT clock_timestamp()').fetchone()[0]
            if run[2] is not None:
                old = conn.execute(sql.SQL('''SELECT status,lease_expires_at FROM {}
                    WHERE conversation_id=%s AND run_id=%s AND attempt_id=%s''').format(s._table('run_attempts')),
                    (conversation_id, run_id, run[2])).fetchone()
                if old is None:
                    raise ValueError('active_attempt_missing')
                if old[0] == 'running' and old[1] > now:
                    return {'acquired': False, 'reason': 'lease_active'}
                conn.execute(sql.SQL('''UPDATE {} SET superseded_at=COALESCE(superseded_at,%s)
                    WHERE conversation_id=%s AND run_id=%s AND attempt_id=%s''').format(s._table('run_attempts')),
                    (now, conversation_id, run_id, run[2]))
            ordinal = conn.execute(sql.SQL('SELECT COALESCE(MAX(attempt_number),0)+1 FROM {} WHERE conversation_id=%s AND run_id=%s')
                .format(s._table('run_attempts')), (conversation_id, run_id)).fetchone()[0]
            attempt_id = uuid4().hex
            conn.execute(sql.SQL('''INSERT INTO {} (conversation_id,run_id,attempt_id,attempt_number,lease_expires_at)
                VALUES (%s,%s,%s,%s,%s + %s * interval '1 second')''').format(s._table('run_attempts')),
                (conversation_id, run_id, attempt_id, ordinal, now, lease_seconds))
            conn.execute(sql.SQL('UPDATE {} SET active_attempt_id=%s WHERE conversation_id=%s AND run_id=%s')
                .format(s._table('runs')), (attempt_id, conversation_id, run_id))
        return {'acquired': True, 'attempt_id': attempt_id, 'attempt_number': ordinal}

    def renew(self, owner_id, conversation_id, run_id, attempt_id, *, lease_seconds=90):
        lease_duration(lease_seconds)
        s = self.store
        with s.connect() as conn:
            conn.execute("SET LOCAL lock_timeout = '2s'")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            version, _ = s._conversation(conn, owner_id, conversation_id)
            updated = conn.execute(sql.SQL('''UPDATE {} a SET lease_expires_at=clock_timestamp() + %s * interval '1 second'
                FROM {} r WHERE a.conversation_id=%s AND a.run_id=%s AND a.attempt_id=%s
                  AND r.conversation_id=a.conversation_id AND r.run_id=a.run_id
                  AND r.active_attempt_id=a.attempt_id AND r.status='running' AND r.input_version=%s
                  AND a.status='running' AND a.lease_expires_at>clock_timestamp()''')
                .format(s._table('run_attempts'), s._table('runs')),
                (lease_seconds, conversation_id, run_id, attempt_id, version))
            return updated.rowcount == 1

    def finish(self, owner_id, conversation_id, run_id, attempt_id, result, *, status='completed'):
        if status not in ('completed', 'failed') or not isinstance(result, dict):
            raise ValueError('invalid_attempt_result')
        s = self.store
        with s.connect() as conn:
            version, current = s._conversation(conn, owner_id, conversation_id)
            run = conn.execute(sql.SQL('SELECT status,active_attempt_id FROM {} WHERE conversation_id=%s AND run_id=%s')
                .format(s._table('runs')), (conversation_id, run_id)).fetchone()
            attempt = conn.execute(sql.SQL('''SELECT status,result,decision,lease_expires_at FROM {}
                WHERE conversation_id=%s AND run_id=%s AND attempt_id=%s''').format(s._table('run_attempts')),
                (conversation_id, run_id, attempt_id)).fetchone()
            if run is None or attempt is None:
                raise ValueError('attempt_not_found')
            if attempt[0] != 'running':
                if attempt[0] != status or attempt[1] != result:
                    raise ValueError('attempt_result_conflict')
                return {'attempt_id': attempt_id, 'decision': attempt[2], 'replayed': True}
            now = conn.execute('SELECT clock_timestamp()').fetchone()[0]
            decision = ('lease_lost' if run[1] != attempt_id or attempt[3] <= now else
                        'run_finished' if run[0] != 'running' else 'accepted')
            conn.execute(sql.SQL('''UPDATE {} SET status=%s,result=%s,finished_at=%s,decision=%s
                WHERE conversation_id=%s AND run_id=%s AND attempt_id=%s''').format(s._table('run_attempts')),
                (status, Jsonb(result), now, decision, conversation_id, run_id, attempt_id))
            publication = None
            if decision == 'accepted':
                publication = s._finish_run(conn, conversation_id, run_id, result, status, version, current)
            return {'attempt_id': attempt_id, 'decision': decision, 'replayed': False, 'publication': publication}


class AttemptHeartbeat:
    """Renew periodically using short connections; the model call holds none."""
    def __init__(self, attempts, owner_id, conversation_id, run_id, attempt_id, lease_seconds):
        self.attempts, self.args = attempts, (owner_id, conversation_id, run_id, attempt_id)
        self.lease_seconds = lease_duration(lease_seconds)
        self.stop = Event()
        self.thread = Thread(target=self._renew, daemon=True)

    def _renew(self):
        while not self.stop.wait(self.lease_seconds / 3):
            try:
                if not self.attempts.renew(*self.args, lease_seconds=self.lease_seconds):
                    return
            except Exception:
                # A failed heartbeat cannot extend ownership. Database fencing
                # remains authoritative even while the provider is still busy.
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
