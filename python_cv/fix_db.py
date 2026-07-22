import sqlite3, os

os.makedirs('outputs', exist_ok=True)
db = 'outputs/warehouse.db'
if os.path.exists(db):
    os.remove(db)
    print('Deleted old DB')

conn = sqlite3.connect(db)

conn.execute('''CREATE TABLE IF NOT EXISTS rfid_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL,
    employee_id TEXT,
    timestamp TEXT NOT NULL,
    device TEXT,
    zone INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)''')

conn.execute('''CREATE TABLE IF NOT EXISTS entry_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    fusion_score REAL,
    zone INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)''')

conn.execute('''CREATE TABLE IF NOT EXISTS anomaly_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    employee_id TEXT,
    track_id INTEGER,
    detail TEXT,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)''')

conn.execute('''CREATE TABLE IF NOT EXISTS presence_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL UNIQUE,
    zone INTEGER NOT NULL,
    track_id INTEGER,
    last_seen TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)''')

conn.commit()

cur = conn.execute('PRAGMA table_info(presence_log)')
cols = [r[1] for r in cur.fetchall()]
print('presence_log columns:', cols)
conn.close()
print('DB created OK')