import argparse
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import DB_PATH, OUTPUTS_DIR


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_domain(domain: str) -> str:
    if not domain:
        return ""
    return str(domain).strip().lower().replace("www.", "")


def load_runs() -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            run_id,
            run_name,
            model,
            region,
            notes,
            created_at
        FROM runs
        ORDER BY run_id ASC;
        """,
        conn,
    )

    conn.close()
    return df


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


def load_answers(run_id: int) -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            answer_id,
            run_id,
            question_id,
            region,
            topic,
            category,
            question,
            ifurniture_mentioned,
            ifurniture_rank,
            ifurniture_sentiment,
            risk_mentioned,
            created_at
        FROM answers
        WHERE run_id = ?
        ORDER BY answer_id ASC;
        """,
        conn,
        params=(run_id,),
    )

    conn.close()
    return df


def load_sources(run_id: int) -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            s.source_id,
            s.answer_id,
            a.run_id,
            s.question_id,
            a.topic,
            a.category,
            a.question,
            s.domain,
            s.url,
            s.source_type,
            s.used_for,
            s.created_at
        FROM sources s
        JOIN answers a
            ON s.answer_id = a.answer_id
        WHERE a.run_id = ?
        ORDER BY s.source_id ASC;
        """,
        conn,
        params=(run_id,),
    )

    conn.close()

    if not df.empty:
        df["domain"] = df["domain"].fillna("").apply(normalize_domain)

    return df


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def avg_rank(series: pd.Series) -> float | None:
    ranks = pd.to_numeric(series, errors="coerce").dropna()

    if ranks.empty:
        return None

    return float(ranks.mean())


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def topic_metrics(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    topic: str,
) -> dict:
    topic_answers = answers_df[answers_df["topic"] == topic].copy()

    n = len(topic_answers)

    if n == 0:
        return {
            "sample_count": 0,
            "visibility_rate": 0.0,
            "first_rate": 0.0,
            "top3_rate": 0.0,
            "risk_rate": 0.0,
            "avg_rank": None,
            "ifurniture_source_coverage": 0.0,
            "ifurniture_source_citations": 0,
            "unique_source_domains": 0,
            "source_domains": "",
        }

    mentioned = topic_answers["ifurniture_mentioned"].fillna(0).astype(int).sum()

    first = (pd.to_numeric(topic_answers["ifurniture_rank"], errors="coerce") == 1).sum()

    top3 = pd.to_numeric(topic_answers["ifurniture_rank"], errors="coerce").apply(
        lambda x: pd.notna(x) and int(x) <= 3
    ).sum()

    risk = topic_answers["risk_mentioned"].fillna(0).astype(int).sum()

    rank_average = avg_rank(topic_answers["ifurniture_rank"])

    if sources_df.empty:
        topic_sources = pd.DataFrame()
    else:
        topic_sources = sources_df[sources_df["topic"] == topic].copy()

    if topic_sources.empty:
        ifurniture_source_citations = 0
        ifurniture_source_coverage = 0.0
        unique_source_domains = 0
        source_domains = ""
    else:
        unique_source_domains = topic_sources["domain"].nunique()
        source_domains = ", ".join(
            sorted(topic_sources["domain"].dropna().astype(str).unique())
        )

        ifurniture_sources = topic_sources[topic_sources["domain"] == "ifurniture.co.nz"]
        ifurniture_source_citations = len(ifurniture_sources)

        questions_with_ifurniture_source = (
            ifurniture_sources["question_id"].dropna().unique()
            if len(ifurniture_sources) > 0
            else []
        )

        ifurniture_source_coverage = safe_rate(
            len(questions_with_ifurniture_source),
            topic_answers["question_id"].nunique(),
        )

    return {
        "sample_count": n,
        "visibility_rate": safe_rate(mentioned, n),
        "first_rate": safe_rate(first, n),
        "top3_rate": safe_rate(top3, n),
        "risk_rate": safe_rate(risk, n),
        "avg_rank": rank_average,
        "ifurniture_source_coverage": ifurniture_source_coverage,
        "ifurniture_source_citations": ifurniture_source_citations,
        "unique_source_domains": unique_source_domains,
        "source_domains": source_domains,
    }


def calculate_lift(before: dict, after: dict) -> dict:
    before_avg_rank = before["avg_rank"]
    after_avg_rank = after["avg_rank"]

    if before_avg_rank is None or after_avg_rank is None:
        avg_rank_change = None
    else:
        # Negative is good because rank 1 is better than rank 4.
        avg_rank_change = after_avg_rank - before_avg_rank

    return {
        "visibility_lift": after["visibility_rate"] - before["visibility_rate"],
        "first_rate_lift": after["first_rate"] - before["first_rate"],
        "top3_rate_lift": after["top3_rate"] - before["top3_rate"],
        "risk_rate_change": after["risk_rate"] - before["risk_rate"],
        "avg_rank_change": avg_rank_change,
        "source_coverage_lift": after["ifurniture_source_coverage"]
        - before["ifurniture_source_coverage"],
        "source_citation_change": after["ifurniture_source_citations"]
        - before["ifurniture_source_citations"],
    }


def judge_impact(lift: dict) -> str:
    score = 0

    if lift["visibility_lift"] > 0:
        score += 2
    elif lift["visibility_lift"] < 0:
        score -= 2

    if lift["top3_rate_lift"] > 0:
        score += 2
    elif lift["top3_rate_lift"] < 0:
        score -= 2

    if lift["first_rate_lift"] > 0:
        score += 1
    elif lift["first_rate_lift"] < 0:
        score -= 1

    if lift["source_coverage_lift"] > 0:
        score += 2
    elif lift["source_coverage_lift"] < 0:
        score -= 2

    if lift["risk_rate_change"] < 0:
        score += 1
    elif lift["risk_rate_change"] > 0:
        score -= 1

    avg_rank_change = lift["avg_rank_change"]

    if avg_rank_change is not None:
        if avg_rank_change < 0:
            score += 2
        elif avg_rank_change > 0:
            score -= 2

    if score >= 4:
        return "strong_positive"
    if score >= 1:
        return "positive_or_promising"
    if score == 0:
        return "no_clear_change"
    if score <= -4:
        return "negative"
    return "mixed_or_uncertain"


def build_action_impact(
    before_run_id: int,
    after_run_id: int,
) -> pd.DataFrame:
    actions_df = load_actions()

    if actions_df.empty:
        raise RuntimeError("No actions found. Please run: python src/action_log.py --import-csv")

    before_answers = load_answers(before_run_id)
    after_answers = load_answers(after_run_id)

    before_sources = load_sources(before_run_id)
    after_sources = load_sources(after_run_id)

    if before_answers.empty:
        raise RuntimeError(f"No answers found for before_run_id={before_run_id}")

    if after_answers.empty:
        raise RuntimeError(f"No answers found for after_run_id={after_run_id}")

    rows = []

    for _, action in actions_df.iterrows():
        topic = action["target_topic"]

        before = topic_metrics(
            answers_df=before_answers,
            sources_df=before_sources,
            topic=topic,
        )

        after = topic_metrics(
            answers_df=after_answers,
            sources_df=after_sources,
            topic=topic,
        )

        lift = calculate_lift(before, after)
        impact_judgement = judge_impact(lift)

        rows.append(
            {
                "action_id": action["action_id"],
                "action_name": action["action_name"],
                "status": action["status"],
                "target_topic": topic,
                "target_source": action["target_source"],
                "expected_impact": action["expected_impact"],
                "publish_date": action["publish_date"],
                "before_run_id": before_run_id,
                "after_run_id": after_run_id,
                "before_sample_count": before["sample_count"],
                "after_sample_count": after["sample_count"],
                "before_visibility": before["visibility_rate"],
                "after_visibility": after["visibility_rate"],
                "visibility_lift": lift["visibility_lift"],
                "before_first_rate": before["first_rate"],
                "after_first_rate": after["first_rate"],
                "first_rate_lift": lift["first_rate_lift"],
                "before_top3_rate": before["top3_rate"],
                "after_top3_rate": after["top3_rate"],
                "top3_rate_lift": lift["top3_rate_lift"],
                "before_risk_rate": before["risk_rate"],
                "after_risk_rate": after["risk_rate"],
                "risk_rate_change": lift["risk_rate_change"],
                "before_avg_rank": before["avg_rank"],
                "after_avg_rank": after["avg_rank"],
                "avg_rank_change": lift["avg_rank_change"],
                "before_source_coverage": before["ifurniture_source_coverage"],
                "after_source_coverage": after["ifurniture_source_coverage"],
                "source_coverage_lift": lift["source_coverage_lift"],
                "before_ifurniture_source_citations": before["ifurniture_source_citations"],
                "after_ifurniture_source_citations": after["ifurniture_source_citations"],
                "source_citation_change": lift["source_citation_change"],
                "before_source_domains": before["source_domains"],
                "after_source_domains": after["source_domains"],
                "impact_judgement": impact_judgement,
            }
        )

    return pd.DataFrame(rows)


def print_impact_summary(df: pd.DataFrame, before_run_id: int, after_run_id: int) -> None:
    print("=" * 110)
    print(f"GEO ACTION IMPACT SUMMARY | before_run_id={before_run_id} | after_run_id={after_run_id}")
    print("=" * 110)

    if df.empty:
        print("No action impact rows found.")
        return

    for _, row in df.iterrows():
        print()
        print(f"Action {row['action_id']} | {row['action_name']}")
        print(f"Topic: {row['target_topic']} | Status: {row['status']} | Judgement: {row['impact_judgement']}")
        print("-" * 110)

        print(
            f"Visibility: {row['before_visibility']:.1%} → {row['after_visibility']:.1%} "
            f"(lift {row['visibility_lift']:+.1%})"
        )

        print(
            f"Top-3 Rate: {row['before_top3_rate']:.1%} → {row['after_top3_rate']:.1%} "
            f"(lift {row['top3_rate_lift']:+.1%})"
        )

        print(
            f"First Rate: {row['before_first_rate']:.1%} → {row['after_first_rate']:.1%} "
            f"(lift {row['first_rate_lift']:+.1%})"
        )

        print(
            f"Risk Rate: {row['before_risk_rate']:.1%} → {row['after_risk_rate']:.1%} "
            f"(change {row['risk_rate_change']:+.1%})"
        )

        print(
            f"Avg Rank: {format_float(row['before_avg_rank'])} → {format_float(row['after_avg_rank'])} "
            f"(change {format_float(row['avg_rank_change'])}; negative is better)"
        )

        print(
            f"iFurniture Source Coverage: {row['before_source_coverage']:.1%} → {row['after_source_coverage']:.1%} "
            f"(lift {row['source_coverage_lift']:+.1%})"
        )

        print(
            f"iFurniture Source Citations: {row['before_ifurniture_source_citations']} → "
            f"{row['after_ifurniture_source_citations']} "
            f"(change {row['source_citation_change']:+})"
        )

        print(f"Before domains: {row['before_source_domains']}")
        print(f"After domains:  {row['after_source_domains']}")


def save_impact_output(df: pd.DataFrame, before_run_id: int, after_run_id: int) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / f"action_impact_before_{before_run_id}_after_{after_run_id}.csv"
    df.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GEO action impact between two runs.")
    parser.add_argument(
        "--before-run-id",
        type=int,
        required=True,
        help="Baseline run ID before actions were implemented.",
    )
    parser.add_argument(
        "--after-run-id",
        type=int,
        required=True,
        help="Comparison run ID after actions were implemented.",
    )

    args = parser.parse_args()

    impact_df = build_action_impact(
        before_run_id=args.before_run_id,
        after_run_id=args.after_run_id,
    )

    print_impact_summary(
        df=impact_df,
        before_run_id=args.before_run_id,
        after_run_id=args.after_run_id,
    )

    output_path = save_impact_output(
        df=impact_df,
        before_run_id=args.before_run_id,
        after_run_id=args.after_run_id,
    )

    print()
    print("Saved output file:")
    print(f"- {output_path}")


if __name__ == "__main__":
    main()