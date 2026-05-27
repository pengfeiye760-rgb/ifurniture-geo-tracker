import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DB_PATH, DATA_DIR
from db import init_db


ACTIONS_CSV = DATA_DIR / "actions.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_actions_csv(path: Path = ACTIONS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Actions CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "action_name",
        "action_type",
        "target_region",
        "target_topic",
        "target_source",
        "expected_impact",
        "status",
        "start_date",
        "publish_date",
        "notes",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"actions.csv is missing columns: {missing_cols}")

    df = df.fillna("")

    return df


def action_exists(
    action_name: str,
    target_region: str,
    target_topic: str,
) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT action_id
        FROM actions
        WHERE action_name = ?
          AND target_region = ?
          AND target_topic = ?
        LIMIT 1;
        """,
        (action_name, target_region, target_topic),
    )

    row = cur.fetchone()
    conn.close()

    return row is not None


def insert_action(row: dict) -> int | None:
    action_name = str(row["action_name"]).strip()
    target_region = str(row["target_region"]).strip()
    target_topic = str(row["target_topic"]).strip()

    if action_exists(action_name, target_region, target_topic):
        return None

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO actions (
            action_name,
            action_type,
            target_region,
            target_topic,
            target_source,
            expected_impact,
            status,
            start_date,
            publish_date,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            action_name,
            str(row["action_type"]).strip(),
            target_region,
            target_topic,
            str(row["target_source"]).strip(),
            str(row["expected_impact"]).strip(),
            str(row["status"]).strip(),
            str(row["start_date"]).strip(),
            str(row["publish_date"]).strip(),
            str(row["notes"]).strip(),
            now_utc(),
        ),
    )

    action_id = cur.lastrowid
    conn.commit()
    conn.close()

    return int(action_id)


def import_actions_from_csv(path: Path = ACTIONS_CSV) -> None:
    init_db()

    df = load_actions_csv(path)

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        action_id = insert_action(row.to_dict())

        if action_id is None:
            skipped += 1
        else:
            inserted += 1
            print(f"Inserted action_id={action_id}: {row['action_name']}")

    print("-" * 80)
    print(f"Import finished. Inserted: {inserted}, Skipped existing: {skipped}")


def load_actions() -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            action_id,
            action_name,
            action_type,
            target_region,
            target_topic,
            target_source,
            expected_impact,
            status,
            start_date,
            publish_date,
            notes,
            created_at
        FROM actions
        ORDER BY action_id ASC;
        """,
        conn,
    )

    conn.close()

    return df


def print_actions() -> None:
    df = load_actions()

    if df.empty:
        print("No actions found.")
        return

    print("=" * 100)
    print("GEO ACTION LOG")
    print("=" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['action_id']:>3} | "
            f"{row['status']:<10} | "
            f"{row['target_topic']:<18} | "
            f"{row['action_name']}"
        )
        print(f"      Type: {row['action_type']}")
        print(f"      Target source: {row['target_source']}")
        print(f"      Expected impact: {row['expected_impact']}")
        print(f"      Start: {row['start_date']} | Publish: {row['publish_date']}")
        print(f"      Notes: {row['notes']}")
        print()


def update_action_status(
    action_id: int,
    status: str,
    publish_date: str = "",
) -> None:
    conn = get_connection()
    cur = conn.cursor()

    if publish_date:
        cur.execute(
            """
            UPDATE actions
            SET status = ?,
                publish_date = ?
            WHERE action_id = ?;
            """,
            (status, publish_date, action_id),
        )
    else:
        cur.execute(
            """
            UPDATE actions
            SET status = ?
            WHERE action_id = ?;
            """,
            (status, action_id),
        )

    conn.commit()
    conn.close()

    print(f"Updated action_id={action_id} to status={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage GEO action log.")
    parser.add_argument(
        "--import-csv",
        action="store_true",
        help="Import actions from data/actions.csv into SQLite.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List actions from SQLite.",
    )
    parser.add_argument(
        "--update-status",
        type=str,
        default="",
        help="Update status for one action. Example: published",
    )
    parser.add_argument(
        "--action-id",
        type=int,
        default=None,
        help="Action ID to update.",
    )
    parser.add_argument(
        "--publish-date",
        type=str,
        default="",
        help="Publish date for the action. Example: 2026-05-28",
    )

    args = parser.parse_args()

    if args.import_csv:
        import_actions_from_csv()

    if args.update_status:
        if args.action_id is None:
            raise ValueError("--action-id is required when using --update-status")

        update_action_status(
            action_id=args.action_id,
            status=args.update_status,
            publish_date=args.publish_date,
        )

    if args.list or not any([args.import_csv, args.update_status]):
        print_actions()


if __name__ == "__main__":
    main()