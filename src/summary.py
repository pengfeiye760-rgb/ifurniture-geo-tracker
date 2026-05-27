import json
import sqlite3
from collections import Counter

from config import DB_PATH


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM answers
        ORDER BY answer_id DESC
        LIMIT 10;
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("No answers found.")
        return

    rows = list(reversed(rows))

    total = len(rows)
    mentioned = sum(row["ifurniture_mentioned"] for row in rows)
    first_rank = sum(1 for row in rows if row["ifurniture_rank"] == 1)
    top3 = sum(
        1
        for row in rows
        if row["ifurniture_rank"] is not None and row["ifurniture_rank"] <= 3
    )
    risk = sum(row["risk_mentioned"] for row in rows)

    sentiment_counter = Counter(row["ifurniture_sentiment"] for row in rows)

    brand_counter = Counter()
    risk_phrase_counter = Counter()

    for row in rows:
        brands = json.loads(row["brands_json"] or "[]")
        brand_counter.update(brands)

        risk_phrases = json.loads(row["risk_phrases_json"] or "[]")
        risk_phrase_counter.update(risk_phrases)

    print("=" * 80)
    print("GEO MVP BASELINE SUMMARY")
    print("=" * 80)
    print(f"Total answers checked: {total}")
    print(f"iFurniture Visibility: {mentioned}/{total} = {mentioned / total:.1%}")
    print(f"First Recommendation Rate: {first_rank}/{total} = {first_rank / total:.1%}")
    print(f"Top-3 Rate: {top3}/{total} = {top3 / total:.1%}")
    print(f"Risk Mention Rate: {risk}/{total} = {risk / total:.1%}")
    print()
    print("Sentiment:")
    for sentiment, count in sentiment_counter.items():
        print(f"  - {sentiment}: {count}")
    print()
    print("Top mentioned brands:")
    for brand, count in brand_counter.most_common(15):
        print(f"  - {brand}: {count}")
    print()
    print("Risk phrases:")
    if risk_phrase_counter:
        for phrase, count in risk_phrase_counter.most_common(15):
            print(f"  - {phrase}: {count}")
    else:
        print("  None")
    print()
    print("Question-level results:")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['question_id']} | "
            f"topic={row['topic']} | "
            f"mentioned={bool(row['ifurniture_mentioned'])} | "
            f"rank={row['ifurniture_rank']} | "
            f"sentiment={row['ifurniture_sentiment']} | "
            f"risk={bool(row['risk_mentioned'])}"
        )

    conn.close()


if __name__ == "__main__":
    main()