import argparse
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from config import DB_PATH, OUTPUTS_DIR


OWNED_DOMAINS = {
    "ifurniture.co.nz",
    "www.ifurniture.co.nz",
}

COMPETITOR_DOMAINS = {
    "ikea.com",
    "ingka.com",
    "bigsave.co.nz",
    "targetfurniture.co.nz",
    "mocka.co.nz",
    "kmart.co.nz",
    "thewarehouse.co.nz",
    "freedomfurniture.co.nz",
    "nood.co.nz",
    "bedpost.co.nz",
    "bedsrus.co.nz",
    "hunterhome.co.nz",
    "trademe.co.nz",
    "ecosa.co.nz",
    "urbansales.co.nz",
    "cintesi.co.nz",
    "kiwihomestore.co.nz",
    "lifestylefurniture.co.nz",
}

MARKETPLACE_DOMAINS = {
    "trademe.co.nz",
    "facebook.com",
}

COMMUNITY_DOMAINS = {
    "reddit.com",
}

MEDIA_GUIDE_DOMAINS = {
    "exploreauckland.nz",
    "viewauckland.co.nz",
    "yourhomeandgarden.co.nz",
    "aucklandtribune.co.nz",
    "theurbanlist.com",
}

AUTHORITY_DOMAINS = {
    "consumer.org.nz",
    "cab.org.nz",
    "salvationarmy.org.nz",
    "habitat.org.nz",
}


def normalize_domain(domain: str) -> str:
    if not domain:
        return ""

    domain = domain.strip().lower()
    domain = domain.replace("www.", "")

    return domain


def clean_url(url: str) -> str:
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return url


def classify_source(domain: str) -> str:
    domain = normalize_domain(domain)

    if domain in OWNED_DOMAINS:
        return "owned_ifurniture"

    if domain in COMMUNITY_DOMAINS:
        return "community"

    if domain in MARKETPLACE_DOMAINS:
        return "marketplace"

    if domain in MEDIA_GUIDE_DOMAINS:
        return "media_or_city_guide"

    if domain in AUTHORITY_DOMAINS:
        return "authority_or_nonprofit"

    if domain in COMPETITOR_DOMAINS:
        return "competitor_or_retailer"

    if domain.endswith(".co.nz") or domain.endswith(".nz"):
        return "nz_local_source"

    return "other"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_run_id() -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT run_id
        FROM runs
        ORDER BY run_id DESC
        LIMIT 1;
        """
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        raise RuntimeError("No runs found. Please run python src/run_batch.py first.")

    return int(row["run_id"])


def load_run(run_id: int) -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM runs
        WHERE run_id = ?;
        """,
        conn,
        params=(run_id,),
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
            s.region,
            a.topic,
            a.category,
            a.question,
            a.ifurniture_mentioned,
            a.ifurniture_rank,
            a.ifurniture_sentiment,
            a.risk_mentioned,
            s.domain,
            s.url,
            s.source_type,
            s.used_for,
            s.sentiment_toward_ifurniture,
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

    if df.empty:
        return df

    df["domain"] = df["domain"].fillna("").apply(normalize_domain)
    df["url"] = df["url"].fillna("").apply(clean_url)
    df["source_group"] = df["domain"].apply(classify_source)

    return df


def build_domain_summary(sources_df: pd.DataFrame) -> pd.DataFrame:
    if sources_df.empty:
        return pd.DataFrame(
            columns=[
                "domain",
                "source_group",
                "citation_count",
                "question_count",
                "topics",
                "question_ids",
                "example_url",
            ]
        )

    grouped = []

    for domain, g in sources_df.groupby("domain"):
        grouped.append(
            {
                "domain": domain,
                "source_group": classify_source(domain),
                "citation_count": len(g),
                "question_count": g["question_id"].nunique(),
                "topics": ", ".join(sorted(g["topic"].dropna().astype(str).unique())),
                "question_ids": ", ".join(sorted(g["question_id"].dropna().astype(str).unique())),
                "example_url": g["url"].dropna().iloc[0] if len(g["url"].dropna()) else "",
            }
        )

    return (
        pd.DataFrame(grouped)
        .sort_values(["question_count", "citation_count", "domain"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def build_question_source_summary(answers_df: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, answer in answers_df.iterrows():
        qid = answer["question_id"]
        q_sources = sources_df[sources_df["question_id"] == qid] if not sources_df.empty else pd.DataFrame()

        domains = []
        source_groups = []

        if not q_sources.empty:
            domains = sorted(q_sources["domain"].dropna().astype(str).unique())
            source_groups = sorted(q_sources["source_group"].dropna().astype(str).unique())

        has_ifurniture_source = any(domain in {"ifurniture.co.nz"} for domain in domains)

        competitor_domains = [
            domain
            for domain in domains
            if classify_source(domain) == "competitor_or_retailer"
        ]

        rows.append(
            {
                "question_id": answer["question_id"],
                "topic": answer["topic"],
                "category": answer["category"],
                "question": answer["question"],
                "ifurniture_mentioned": bool(answer["ifurniture_mentioned"]),
                "ifurniture_rank": answer["ifurniture_rank"],
                "ifurniture_sentiment": answer["ifurniture_sentiment"],
                "risk_mentioned": bool(answer["risk_mentioned"]),
                "source_count": len(q_sources),
                "unique_domain_count": len(domains),
                "has_ifurniture_source": has_ifurniture_source,
                "domains": ", ".join(domains),
                "source_groups": ", ".join(source_groups),
                "competitor_domains": ", ".join(competitor_domains),
            }
        )

    return pd.DataFrame(rows)


def build_opportunity_summary(question_df: pd.DataFrame) -> pd.DataFrame:
    if question_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in question_df.iterrows():
        reasons = []
        priority_score = 0

        if not row["ifurniture_mentioned"]:
            reasons.append("iFurniture not mentioned")
            priority_score += 5

        rank = row["ifurniture_rank"]

        if pd.notna(rank):
            try:
                rank_int = int(rank)

                if rank_int > 3:
                    reasons.append(f"iFurniture rank is outside Top 3: {rank_int}")
                    priority_score += 3
                elif rank_int == 1:
                    reasons.append("Strong position: iFurniture ranks #1")
                    priority_score -= 2

            except Exception:
                pass

        if not row["has_ifurniture_source"]:
            reasons.append("No ifurniture.co.nz source cited")
            priority_score += 4

        if row["competitor_domains"]:
            reasons.append(f"Competitor sources cited: {row['competitor_domains']}")
            priority_score += 1

        if row["risk_mentioned"]:
            reasons.append("Risk phrase mentioned")
            priority_score += 3

        if priority_score > 0:
            rows.append(
                {
                    "question_id": row["question_id"],
                    "topic": row["topic"],
                    "question": row["question"],
                    "ifurniture_mentioned": row["ifurniture_mentioned"],
                    "ifurniture_rank": row["ifurniture_rank"],
                    "has_ifurniture_source": row["has_ifurniture_source"],
                    "priority_score": priority_score,
                    "opportunity_reason": " | ".join(reasons),
                    "recommended_action_type": recommend_action(row),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "question_id",
                "topic",
                "question",
                "ifurniture_mentioned",
                "ifurniture_rank",
                "has_ifurniture_source",
                "priority_score",
                "opportunity_reason",
                "recommended_action_type",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["priority_score", "question_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def recommend_action(row: pd.Series) -> str:
    topic = str(row.get("topic", "")).lower()
    question = str(row.get("question", "")).lower()

    if "delivery" in topic or "delivery" in question:
        return "Create / improve Logistics Transparency and Delivery Reliability page"

    if "small" in topic or "sofa" in question or "living room" in question:
        return "Create Small Apartment Sofa / Small Living Room Sofa guide"

    if "first_home" in topic or "first-home" in question or "first home" in question:
        return "Create First-home Buyer Furniture Guide"

    if "dining" in topic or "dining table" in question:
        return "Create Affordable Dining Tables NZ category guide"

    if "ikea" in topic or "ikea" in question:
        return "Create IKEA Alternative NZ comparison page"

    if "showroom" in topic or "showroom" in question:
        return "Improve showroom page with structured facts and schema"

    if "rental" in topic or "airbnb" in question or "property" in question:
        return "Strengthen rental property / Airbnb package pages"

    return "Create topic-specific GEO landing page and secure third-party citations"


def print_console_summary(
    run_id: int,
    run_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    domain_summary_df: pd.DataFrame,
    question_summary_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
) -> None:
    total_questions = len(answers_df)
    total_sources = len(sources_df)
    unique_domains = sources_df["domain"].nunique() if not sources_df.empty else 0

    visibility = answers_df["ifurniture_mentioned"].fillna(0).astype(int).sum() / total_questions if total_questions else 0
    first_rate = (answers_df["ifurniture_rank"] == 1).sum() / total_questions if total_questions else 0
    top3_rate = answers_df["ifurniture_rank"].apply(
        lambda x: pd.notna(x) and int(x) <= 3
    ).sum() / total_questions if total_questions else 0
    risk_rate = answers_df["risk_mentioned"].fillna(0).astype(int).sum() / total_questions if total_questions else 0

    ifurniture_source_coverage = (
        question_summary_df["has_ifurniture_source"].sum() / total_questions
        if total_questions and not question_summary_df.empty
        else 0
    )

    print("=" * 90)
    print(f"GEO SOURCE SUMMARY | run_id={run_id}")
    print("=" * 90)

    if not run_df.empty:
        run = run_df.iloc[0]
        print(f"Run name: {run.get('run_name', '')}")
        print(f"Model: {run.get('model', '')}")
        print(f"Region: {run.get('region', '')}")
        print(f"Created at: {run.get('created_at', '')}")
        print(f"Notes: {run.get('notes', '')}")
        print("-" * 90)

    print(f"Questions: {total_questions}")
    print(f"Total source citations: {total_sources}")
    print(f"Unique source domains: {unique_domains}")
    print(f"iFurniture Visibility: {visibility:.1%}")
    print(f"First Recommendation Rate: {first_rate:.1%}")
    print(f"Top-3 Rate: {top3_rate:.1%}")
    print(f"Risk Mention Rate: {risk_rate:.1%}")
    print(f"ifurniture.co.nz Source Coverage: {ifurniture_source_coverage:.1%}")
    print()

    print("Top Source Domains:")
    print("-" * 90)
    if domain_summary_df.empty:
        print("No source domains found.")
    else:
        for _, row in domain_summary_df.head(20).iterrows():
            print(
                f"{row['domain']:35} | "
                f"group={row['source_group']:22} | "
                f"questions={row['question_count']:2} | "
                f"citations={row['citation_count']:2} | "
                f"topics={row['topics']}"
            )

    print()
    print("Question-Level Source Coverage:")
    print("-" * 90)
    if question_summary_df.empty:
        print("No question-level source summary found.")
    else:
        for _, row in question_summary_df.iterrows():
            print(
                f"{row['question_id']} | "
                f"topic={row['topic']:18} | "
                f"mentioned={row['ifurniture_mentioned']} | "
                f"rank={row['ifurniture_rank']} | "
                f"ifurniture_source={row['has_ifurniture_source']} | "
                f"sources={row['unique_domain_count']} | "
                f"domains={row['domains']}"
            )

    print()
    print("Priority Opportunities:")
    print("-" * 90)
    if opportunity_df.empty:
        print("No priority opportunities found.")
    else:
        for _, row in opportunity_df.head(10).iterrows():
            print(
                f"{row['question_id']} | "
                f"score={row['priority_score']} | "
                f"topic={row['topic']} | "
                f"action={row['recommended_action_type']}"
            )
            print(f"  Reason: {row['opportunity_reason']}")
            print(f"  Question: {row['question']}")
            print()


def save_outputs(
    run_id: int,
    domain_summary_df: pd.DataFrame,
    question_summary_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    domain_path = OUTPUTS_DIR / f"source_domain_summary_run_{run_id}.csv"
    question_path = OUTPUTS_DIR / f"question_source_summary_run_{run_id}.csv"
    opportunity_path = OUTPUTS_DIR / f"source_opportunities_run_{run_id}.csv"

    domain_summary_df.to_csv(domain_path, index=False)
    question_summary_df.to_csv(question_path, index=False)
    opportunity_df.to_csv(opportunity_path, index=False)

    print()
    print("Saved output files:")
    print(f"- {domain_path}")
    print(f"- {question_path}")
    print(f"- {opportunity_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise GEO source citations for a specific run.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID to analyse. If omitted, the latest run will be used.",
    )

    args = parser.parse_args()

    run_id = args.run_id or get_latest_run_id()

    run_df = load_run(run_id)
    answers_df = load_answers(run_id)
    sources_df = load_sources(run_id)

    if answers_df.empty:
        raise RuntimeError(f"No answers found for run_id={run_id}.")

    domain_summary_df = build_domain_summary(sources_df)
    question_summary_df = build_question_source_summary(answers_df, sources_df)
    opportunity_df = build_opportunity_summary(question_summary_df)

    print_console_summary(
        run_id=run_id,
        run_df=run_df,
        answers_df=answers_df,
        sources_df=sources_df,
        domain_summary_df=domain_summary_df,
        question_summary_df=question_summary_df,
        opportunity_df=opportunity_df,
    )

    save_outputs(
        run_id=run_id,
        domain_summary_df=domain_summary_df,
        question_summary_df=question_summary_df,
        opportunity_df=opportunity_df,
    )


if __name__ == "__main__":
    main()