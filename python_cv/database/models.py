CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS rfid_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uid         TEXT    NOT NULL,
    employee_id TEXT,
    timestamp   TEXT    NOT NULL,
    device      TEXT,
    zone        INTEGER,
    created_at  TEXT    DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS entry_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id  TEXT    NOT NULL,
    track_id     TEXT    NOT NULL,
    entry_time   TEXT    NOT NULL,
    fusion_score REAL,
    zone         INTEGER DEFAULT 1,
    created_at   TEXT    DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS anomaly_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT    NOT NULL,
    employee_id TEXT,
    track_id    TEXT,
    detail      TEXT,
    timestamp   TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS presence_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT    NOT NULL UNIQUE,
    zone        INTEGER NOT NULL,
    track_id    TEXT,
    current_zone INTEGER,
    track_key   TEXT,
    source      TEXT,
    status      TEXT    DEFAULT 'active',
    first_seen  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    last_seen   TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    created_at  TEXT    DEFAULT (datetime('now','localtime'))
);
"""
