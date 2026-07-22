# python_cv/database/database.py

import sqlite3
import threading
from datetime import datetime

from database.models import CREATE_TABLES
from config.settings import DB_PATH


class Database:
    def __init__(self):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(CREATE_TABLES)
            self._migrate_schema_locked()
            self._conn.commit()

    def _table_columns_locked(self, table_name: str) -> set[str]:
        cur = self._conn.execute(f"PRAGMA table_info({table_name})")
        return {row["name"] for row in cur.fetchall()}

    def _migrate_schema_locked(self):
        """
        Migration nhẹ cho DB cũ.
        SQLite không cho ADD COLUMN với DEFAULT dạng datetime(...),
        nên thêm cột trước, sau đó UPDATE giá trị bằng câu SQL riêng.
        """
        presence_cols = self._table_columns_locked("presence_log")

        if "track_id" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN track_id TEXT")

        if "first_seen" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN first_seen TEXT")
            self._conn.execute(
                """
                UPDATE presence_log
                SET first_seen = COALESCE(created_at, last_seen, datetime('now','localtime'))
                WHERE first_seen IS NULL
                """
            )

        if "last_seen" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN last_seen TEXT")
            self._conn.execute(
                """
                UPDATE presence_log
                SET last_seen = COALESCE(created_at, first_seen, datetime('now','localtime'))
                WHERE last_seen IS NULL
                """
            )

        if "current_zone" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN current_zone INTEGER")

        if "track_key" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN track_key TEXT")

        if "source" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN source TEXT")

        if "status" not in presence_cols:
            self._conn.execute("ALTER TABLE presence_log ADD COLUMN status TEXT DEFAULT 'active'")

        self._conn.execute(
            """
            UPDATE presence_log
            SET current_zone = COALESCE(current_zone, zone)
            WHERE current_zone IS NULL
            """
        )
        self._conn.execute(
            """
            UPDATE presence_log
            SET track_key = COALESCE(track_key, track_id)
            WHERE track_key IS NULL
            """
        )
        self._conn.execute(
            """
            UPDATE presence_log
            SET source = COALESCE(source, 'legacy')
            WHERE source IS NULL
            """
        )
        self._conn.execute(
            """
            UPDATE presence_log
            SET status = COALESCE(status, 'active')
            WHERE status IS NULL
            """
        )

        self._conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_presence_employee_id
            ON presence_log(employee_id)
            """
        )
    # ── RFID event ────────────────────────────────────────────────────────────
    def log_rfid_event(
        self,
        uid: str,
        employee_id: str | None,
        timestamp: str,
        device: str = "",
        zone: int = 1,
    ):
        sql = """
        INSERT INTO rfid_events (uid, employee_id, timestamp, device, zone)
        VALUES (?, ?, ?, ?, ?)
        """
        self._exec(sql, (uid, employee_id, timestamp, device, zone))

    def log_unknown_uid(self, uid: str, timestamp: datetime):
        self.log_rfid_event(uid, None, timestamp.isoformat(), zone=1)

    # ── Entry confirm ─────────────────────────────────────────────────────────
    def log_entry(
        self,
        employee_id: str,
        track_id,
        entry_time: datetime,
        fusion_score: float,
        zone: int = 1,
    ):
        sql = """
        INSERT INTO entry_log (employee_id, track_id, entry_time, fusion_score, zone)
        VALUES (?, ?, ?, ?, ?)
        """
        self._exec(
            sql,
            (
                employee_id,
                str(track_id),
                entry_time.isoformat(),
                float(fusion_score),
                int(zone),
            ),
        )

    # ── Anomaly ───────────────────────────────────────────────────────────────
    def log_anomaly(
        self,
        atype: str,
        employee_id: str | None,
        track_id,
        detail: str,
        timestamp: datetime,
    ):
        sql = """
        INSERT INTO anomaly_log (type, employee_id, track_id, detail, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """
        self._exec(
            sql,
            (
                atype,
                employee_id,
                None if track_id is None else str(track_id),
                detail,
                timestamp.isoformat(),
            ),
        )

    # ── Presence update ───────────────────────────────────────────────────────
    def update_presence(
        self,
        employee_id: str,
        current_zone: int,
        track_key,
        source: str = "cv",
        status: str = "active",
    ):
        """
        Cập nhật vị trí hiện tại của nhân viên.
        track_id hiện dùng dạng zone-aware, ví dụ: zone2:1, zone3:4.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_zone = int(current_zone)
        track_key = str(track_key)

        sql = """
        INSERT INTO presence_log (
            employee_id, zone, track_id, current_zone, track_key,
            source, status, first_seen, last_seen
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(employee_id) DO UPDATE SET
            zone=excluded.zone,
            track_id=excluded.track_id,
            current_zone=excluded.current_zone,
            track_key=excluded.track_key,
            source=excluded.source,
            status=excluded.status,
            last_seen=excluded.last_seen
        """
        self._exec(
            sql,
            (
                employee_id,
                current_zone,
                track_key,
                current_zone,
                track_key,
                source,
                status,
                now,
                now,
            ),
        )

    def remove_presence(self, employee_id: str):
        self._exec("DELETE FROM presence_log WHERE employee_id=?", (employee_id,))

    # ── Dashboard queries ─────────────────────────────────────────────────────
    def get_today_entries(self) -> list:
        sql = """
        SELECT employee_id, track_id, entry_time, zone, fusion_score
        FROM entry_log
        WHERE date(entry_time) = date('now','localtime')
        ORDER BY entry_time DESC
        """
        return self._query(sql)

    def get_recent_anomalies(self, limit: int = 20) -> list:
        sql = """
        SELECT type, employee_id, track_id, detail, timestamp
        FROM anomaly_log
        ORDER BY created_at DESC
        LIMIT ?
        """
        return self._query(sql, (limit,))

    def get_rfid_events(self, limit: int = 50) -> list:
        sql = """
        SELECT uid, employee_id, timestamp, device, zone
        FROM rfid_events
        ORDER BY created_at DESC
        LIMIT ?
        """
        return self._query(sql, (limit,))

    def get_current_presence(self) -> list:
        sql = """
        SELECT
            employee_id,
            current_zone,
            track_key,
            source,
            status,
            first_seen,
            last_seen,
            current_zone AS zone,
            track_key AS track_id
        FROM presence_log
        WHERE status = 'active'
          AND last_seen >= datetime('now','localtime','-30 seconds')
        ORDER BY last_seen DESC
        """
        return self._query(sql)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def close(self):
        self._conn.close()
