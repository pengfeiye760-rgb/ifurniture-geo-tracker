import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config import DB_PATH, DATA_DIR, QUESTIONS_CSV


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT,
            model TEXT,
            region TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
            answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            question_id TEXT NOT NULL,
            region TEXT,
            topic TEXT,
            category TEXT,
            question TEXT NOT NULL,

            model TEXT,
            raw_answer TEXT,

            ifurniture_mentioned INTEGER,
            ifurniture_rank INTEGER,
            ifurniture_sentiment TEXT,
            risk_mentioned INTEGER,
            risk_phrases_json TEXT,
            brands_json TEXT,
            sources_json TEXT,

            created_at TEXT NOT NULL,

            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            answer_id INTEGER NOT NULL,
            question_id TEXT NOT NULL,
            region TEXT,
            domain TEXT,
            url TEXT,
            source_type TEXT,
            used_for TEXT,
            sentiment_toward_ifurniture TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY (answer_id) REFERENCES answers(answer_id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_name TEXT NOT NULL,
            action_type TEXT,
            target_region TEXT,
            target_topic TEXT,
            target_source TEXT,
            expected_impact TEXT,
            status TEXT,
            start_date TEXT,
            publish_date TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


def create_run(run_name: str, model: str, region: str = "NZ", notes: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO runs (run_name, model, region, notes, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (run_name, model, region, notes, now_utc()),
    )

    run_id = cur.lastrowid
    conn.commit()
    conn.close()

    return int(run_id)


def insert_answer(
    run_id: int,
    question_row: Dict[str, Any],
    model: str,
    raw_answer: str,
    features: Dict[str, Any],
) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO answers (
            run_id,
            question_id,
            region,
            topic,
            category,
            question,
            model,
            raw_answer,
            ifurniture_mentioned,
            ifurniture_rank,
            ifurniture_sentiment,
            risk_mentioned,
            risk_phrases_json,
            brands_json,
            sources_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            run_id,
            question_row["question_id"],
            question_row["region"],
            question_row["topic"],
            question_row["category"],
            question_row["question"],
            model,
            raw_answer,
            int(features.get("ifurniture_mentioned", False)),
            features.get("ifurniture_rank"),
            features.get("ifurniture_sentiment"),
            int(features.get("risk_mentioned", False)),
            json.dumps(features.get("risk_phrases", []), ensure_ascii=False),
            json.dumps(features.get("brands", []), ensure_ascii=False),
            json.dumps(features.get("sources", []), ensure_ascii=False),
            now_utc(),
        ),
    )

    answer_id = cur.lastrowid
    conn.commit()
    conn.close()

    return int(answer_id)


def insert_sources(
    answer_id: int,
    question_id: str,
    region: str,
    sources: list[Dict[str, Any]],
) -> None:
    if not sources:
        return

    conn = get_connection()
    cur = conn.cursor()

    for source in sources:
        cur.execute(
            """
            INSERT INTO sources (
                answer_id,
                question_id,
                region,
                domain,
                url,
                source_type,
                used_for,
                sentiment_toward_ifurniture,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                answer_id,
                question_id,
                region,
                source.get("domain", ""),
                source.get("url", ""),
                source.get("source_type", "unknown"),
                source.get("used_for", ""),
                source.get("sentiment_toward_ifurniture", "unknown"),
                now_utc(),
            ),
        )

    conn.commit()
    conn.close()


def check_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row["name"] for row in cur.fetchall()]

    print("DB_PATH:", DB_PATH)
    print("DB exists:", Path(DB_PATH).exists())
    print("Tables:", tables)

    conn.close()


if __name__ == "__main__":
    print("QUESTIONS_CSV exists:", QUESTIONS_CSV.exists())
    init_db()
    check_db()