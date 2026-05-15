#!/usr/bin/env python3
"""SQLite-backed persistent event store with time-range queries."""
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class PersistentEventStore:
    """Thread-safe SQLite event store with automatic rotation."""

    def __init__(self, db_path: str = "output/events.db", max_age_days: int = 30):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._max_age_days = max_age_days
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT,
                    event_type TEXT NOT NULL,
                    track_id INTEGER,
                    timestamp TEXT NOT NULL,
                    epoch_seconds REAL NOT NULL,
                    zone_name TEXT,
                    line_name TEXT,
                    dwell_seconds REAL,
                    bbox TEXT,
                    centroid TEXT,
                    meta TEXT,
                    created_at REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(epoch_seconds)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
            """)
            conn.commit()

    def add(self, event: Dict) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO events
                (camera_id, camera_name, event_type, track_id, timestamp, epoch_seconds,
                 zone_name, line_name, dwell_seconds, bbox, centroid, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("camera_id", ""),
                    event.get("camera_name", ""),
                    event.get("event_type", ""),
                    event.get("track_id"),
                    event.get("timestamp", ""),
                    event.get("epoch_seconds", 0.0),
                    event.get("zone_name"),
                    event.get("line_name"),
                    event.get("dwell_seconds"),
                    json.dumps(event.get("bbox")) if "bbox" in event else None,
                    json.dumps(event.get("centroid")) if "centroid" in event else None,
                    json.dumps({k: v for k, v in event.items()
                                if k not in ("camera_id", "camera_name", "event_type",
                                            "track_id", "timestamp", "epoch_seconds",
                                            "zone_name", "line_name", "dwell_seconds",
                                            "bbox", "centroid")}) or None,
                ),
            )
            conn.commit()

    def recent(self, n: int = 50, cam_id: str = None) -> List[Dict]:
        with self._lock:
            conn = self._conn()
            if cam_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE camera_id = ? ORDER BY epoch_seconds DESC LIMIT ?",
                    (cam_id, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY epoch_seconds DESC LIMIT ?",
                    (n,),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_events_by_date(self, date_str: str, limit: int = 100, cam_id: str = None) -> List[Dict]:
        if not date_str:
            return self.recent(limit, cam_id)
        with self._lock:
            conn = self._conn()
            pattern = f"{date_str}%"
            if cam_id:
                rows = conn.execute(
                    """SELECT * FROM events
                       WHERE camera_id = ? AND timestamp LIKE ?
                       ORDER BY epoch_seconds DESC LIMIT ?""",
                    (cam_id, pattern, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM events
                       WHERE timestamp LIKE ?
                       ORDER BY epoch_seconds DESC LIMIT ?""",
                    (pattern, limit),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_events_range(self, start: float, end: float, cam_id: str = None, limit: int = 1000) -> List[Dict]:
        with self._lock:
            conn = self._conn()
            if cam_id:
                rows = conn.execute(
                    """SELECT * FROM events
                       WHERE camera_id = ? AND epoch_seconds BETWEEN ? AND ?
                       ORDER BY epoch_seconds DESC LIMIT ?""",
                    (cam_id, start, end, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM events
                       WHERE epoch_seconds BETWEEN ? AND ?
                       ORDER BY epoch_seconds DESC LIMIT ?""",
                    (start, end, limit),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def count(self, cam_id: str = None) -> int:
        with self._lock:
            conn = self._conn()
            if cam_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE camera_id = ?",
                    (cam_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            return row[0] if row else 0

    def cleanup_old(self) -> int:
        cutoff = time.time() - self._max_age_days * 86400
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM events WHERE epoch_seconds < ?", (cutoff,))
            conn.commit()
            return cur.rowcount

    def clear(self, cam_id: str = None) -> None:
        with self._lock:
            conn = self._conn()
            if cam_id:
                conn.execute("DELETE FROM events WHERE camera_id = ?", (cam_id,))
            else:
                conn.execute("DELETE FROM events")
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        d = dict(row)
        for key in ("bbox", "centroid"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        if d.get("meta"):
            try:
                meta = json.loads(d["meta"])
                d.update(meta)
            except Exception:
                pass
            del d["meta"]
        return d

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None