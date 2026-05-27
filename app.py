import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import DB_PATH


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


st.set_page_config(
    page_title="iFurniture GEO Performance Tracker",
    page_icon="📊",
    layout="wide",
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_domain(domain: str) -> str:
    if not domain:
        return ""
    return domain.strip().lower().replace("www.", "")


def classify_source(domain: str) -> str:
    domain = normalize_domain(domain)

    if domain in {"ifurniture.co.nz"}:
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


def safe_json_loads(value, default):
    try:
        if value is None or value == "":
            return default
        return json.loads(value)
    except Exception:
        return default


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


def load_runs() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT run_id, run_name, model, region, notes, created_at
        FROM runs
        ORDER BY run_id DESC;
        """,
        conn,
    )
    conn.close()
    return df


def load_answers(run_id: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT *
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
    df["source_group"] = df["domain"].apply(classify_source)

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


def calculate_metrics(answers_df: pd.DataFrame, sources_df: pd.DataFrame) -> dict:
    total = len(answers_df)

    if total == 0:
        return {
            "total": 0,
            "visibility": 0,
            "first_rate": 0,
            "top3_rate": 0,
            "risk_rate": 0,
            "ifurniture_source_coverage": 0,
            "total_sources": 0,
            "unique_domains": 0,
        }

    mentioned = answers_df["ifurniture_mentioned"].fillna(0).astype(int).sum()
    first = (answers_df["ifurniture_rank"] == 1).sum()

    top3 = answers_df["ifurniture_rank"].apply(
        lambda x: pd.notna(x) and int(x) <= 3
    ).sum()

    risk = answers_df["risk_mentioned"].fillna(0).astype(int).sum()

    if sources_df.empty:
        ifurniture_source_coverage = 0
        total_sources = 0
        unique_domains = 0
    else:
        question_has_ifurniture = (
            sources_df[sources_df["domain"] == "ifurniture.co.nz"]["question_id"]
            .dropna()
            .unique()
        )
        ifurniture_source_coverage = len(question_has_ifurniture) / total
        total_sources = len(sources_df)
        unique_domains = sources_df["domain"].nunique()

    return {
        "total": total,
        "visibility": mentioned / total,
        "first_rate": first / total,
        "top3_rate": top3 / total,
        "risk_rate": risk / total,
        "ifurniture_source_coverage": ifurniture_source_coverage,
        "total_sources": total_sources,
        "unique_domains": unique_domains,
    }


def build_brand_count(answers_df: pd.DataFrame) -> pd.DataFrame:
    counts = {}

    for value in answers_df["brands_json"].fillna("[]"):
        brands = safe_json_loads(value, [])
        for brand in brands:
            counts[brand] = counts.get(brand, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["brand", "count"])

    return (
        pd.DataFrame(
            [{"brand": brand, "count": count} for brand, count in counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def build_risk_count(answers_df: pd.DataFrame) -> pd.DataFrame:
    counts = {}

    for value in answers_df["risk_phrases_json"].fillna("[]"):
        phrases = safe_json_loads(value, [])
        for phrase in phrases:
            counts[phrase] = counts.get(phrase, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["risk_phrase", "count"])

    return (
        pd.DataFrame(
            [{"risk_phrase": phrase, "count": count} for phrase, count in counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


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

    rows = []

    for domain, g in sources_df.groupby("domain"):
        rows.append(
            {
                "domain": domain,
                "source_group": classify_source(domain),
                "citation_count": len(g),
                "question_count": g["question_id"].nunique(),
                "topics": ", ".join(sorted(g["topic"].dropna().astype(str).unique())),
                "question_ids": ", ".join(
                    sorted(g["question_id"].dropna().astype(str).unique())
                ),
                "example_url": g["url"].dropna().iloc[0]
                if len(g["url"].dropna())
                else "",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["question_count", "citation_count", "domain"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def build_question_source_summary(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, answer in answers_df.iterrows():
        qid = answer["question_id"]

        if sources_df.empty:
            q_sources = pd.DataFrame()
        else:
            q_sources = sources_df[sources_df["question_id"] == qid]

        domains = []
        source_groups = []

        if not q_sources.empty:
            domains = sorted(q_sources["domain"].dropna().astype(str).unique())
            source_groups = sorted(q_sources["source_group"].dropna().astype(str).unique())

        has_ifurniture_source = "ifurniture.co.nz" in domains

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
    ranks = pd.to_numeric(topic_answers["ifurniture_rank"], errors="coerce")

    first = (ranks == 1).sum()
    top3 = ranks.apply(lambda x: pd.notna(x) and int(x) <= 3).sum()
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
    actions_df: pd.DataFrame,
    before_answers: pd.DataFrame,
    before_sources: pd.DataFrame,
    after_answers: pd.DataFrame,
    after_sources: pd.DataFrame,
    before_run_id: int,
    after_run_id: int,
) -> pd.DataFrame:
    if actions_df.empty:
        return pd.DataFrame()

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


def download_button_for_df(label: str, df: pd.DataFrame, file_name: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def main():
    st.title("📊 iFurniture GEO Performance Tracker")
    st.caption(
        "MVP dashboard: AI visibility, recommendation rank, source coverage, priority GEO opportunities, action tracking, and impact comparison."
    )

    if not Path(DB_PATH).exists():
        st.error("Database not found. Please run: python src/run_batch.py")
        return

    runs_df = load_runs()

    if runs_df.empty:
        st.warning("No runs found. Please run: python src/run_batch.py")
        return

    st.sidebar.header("Run Selection")

    run_options = {
        f"Run {row.run_id} | {row.created_at} | {row.model}": row.run_id
        for row in runs_df.itertuples()
    }

    selected_label = st.sidebar.selectbox(
        "Select a run",
        options=list(run_options.keys()),
    )

    selected_run_id = run_options[selected_label]
    selected_run = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]

    answers_df = load_answers(selected_run_id)
    sources_df = load_sources(selected_run_id)

    st.sidebar.markdown("### Run Info")
    st.sidebar.write(f"**Run ID:** {selected_run['run_id']}")
    st.sidebar.write(f"**Model:** {selected_run['model']}")
    st.sidebar.write(f"**Region:** {selected_run['region']}")
    st.sidebar.write(f"**Created:** {selected_run['created_at']}")
    st.sidebar.write(f"**Notes:** {selected_run['notes']}")

    if answers_df.empty:
        st.warning("No answers found for this run.")
        return

    metrics = calculate_metrics(answers_df, sources_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions", metrics["total"])
    col2.metric("AI Visibility", f"{metrics['visibility']:.0%}")
    col3.metric("First Recommendation", f"{metrics['first_rate']:.0%}")
    col4.metric("Top-3 Rate", f"{metrics['top3_rate']:.0%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Risk Mention", f"{metrics['risk_rate']:.0%}")
    col6.metric("iFurniture Source Coverage", f"{metrics['ifurniture_source_coverage']:.0%}")
    col7.metric("Source Citations", metrics["total_sources"])
    col8.metric("Unique Domains", metrics["unique_domains"])

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview",
            "Source Intelligence",
            "Priority Opportunities",
            "Raw Answers",
            "Action Tracker",
            "Action Impact",
        ]
    )

    with tab1:
        st.subheader("Question-Level GEO Results")

        display_df = answers_df[
            [
                "question_id",
                "region",
                "topic",
                "category",
                "ifurniture_mentioned",
                "ifurniture_rank",
                "ifurniture_sentiment",
                "risk_mentioned",
                "question",
            ]
        ].copy()

        display_df["ifurniture_mentioned"] = display_df["ifurniture_mentioned"].astype(bool)
        display_df["risk_mentioned"] = display_df["risk_mentioned"].astype(bool)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Top Mentioned Brands")
            brand_df = build_brand_count(answers_df)

            if brand_df.empty:
                st.info("No brand data found.")
            else:
                st.bar_chart(brand_df.set_index("brand")["count"])
                st.dataframe(brand_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("Risk Phrases")
            risk_df = build_risk_count(answers_df)

            if risk_df.empty:
                st.success("No risk phrases found in this run.")
            else:
                st.bar_chart(risk_df.set_index("risk_phrase")["count"])
                st.dataframe(risk_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Top Source Domains")

        domain_summary_df = build_domain_summary(sources_df)

        if domain_summary_df.empty:
            st.info("No source citations found. Try running with use_web_search=True.")
        else:
            chart_df = domain_summary_df.head(15).set_index("domain")["question_count"]
            st.bar_chart(chart_df)

            st.dataframe(
                domain_summary_df,
                use_container_width=True,
                hide_index=True,
            )

            download_button_for_df(
                "Download Source Domain Summary CSV",
                domain_summary_df,
                f"source_domain_summary_run_{selected_run_id}.csv",
            )

        st.divider()

        st.subheader("Question-Level Source Coverage")

        question_source_df = build_question_source_summary(answers_df, sources_df)

        if question_source_df.empty:
            st.info("No question-level source summary found.")
        else:
            st.dataframe(
                question_source_df,
                use_container_width=True,
                hide_index=True,
            )

            download_button_for_df(
                "Download Question Source Summary CSV",
                question_source_df,
                f"question_source_summary_run_{selected_run_id}.csv",
            )

    with tab3:
        st.subheader("Priority GEO Opportunities")

        question_source_df = build_question_source_summary(answers_df, sources_df)
        opportunity_df = build_opportunity_summary(question_source_df)

        if opportunity_df.empty:
            st.success("No priority opportunities found in this run.")
        else:
            top_priority = opportunity_df.head(5)

            for _, row in top_priority.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"### {row['question_id']} | {row['topic']} | Priority Score: {row['priority_score']}"
                    )
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**Recommended Action:** {row['recommended_action_type']}")
                    st.write(f"**Reason:** {row['opportunity_reason']}")

            st.divider()

            st.dataframe(
                opportunity_df,
                use_container_width=True,
                hide_index=True,
            )

            download_button_for_df(
                "Download Priority Opportunities CSV",
                opportunity_df,
                f"source_opportunities_run_{selected_run_id}.csv",
            )

    with tab4:
        st.subheader("Raw AI Answers")

        question_labels = {
            f"{row.question_id} | {row.topic} | rank={row.ifurniture_rank} | {row.question[:70]}": row.answer_id
            for row in answers_df.itertuples()
        }

        selected_question_label = st.selectbox(
            "Select an answer to inspect",
            options=list(question_labels.keys()),
        )

        selected_answer_id = question_labels[selected_question_label]
        selected_answer = answers_df[
            answers_df["answer_id"] == selected_answer_id
        ].iloc[0]

        st.markdown("### Question")
        st.write(selected_answer["question"])

        st.markdown("### AI Answer")
        st.write(selected_answer["raw_answer"])

        st.markdown("### Extracted Features")

        extracted = {
            "ifurniture_mentioned": bool(selected_answer["ifurniture_mentioned"]),
            "ifurniture_rank": selected_answer["ifurniture_rank"],
            "ifurniture_sentiment": selected_answer["ifurniture_sentiment"],
            "risk_mentioned": bool(selected_answer["risk_mentioned"]),
            "risk_phrases": safe_json_loads(selected_answer["risk_phrases_json"], []),
            "brands": safe_json_loads(selected_answer["brands_json"], []),
            "sources": safe_json_loads(selected_answer["sources_json"], []),
        }

        st.json(extracted)

        st.markdown("### Cited Sources for This Answer")

        if sources_df.empty:
            st.info("No source citations found for this run.")
        else:
            q_sources = sources_df[sources_df["answer_id"] == selected_answer_id]

            if q_sources.empty:
                st.info("No source citations found for this answer.")
            else:
                st.dataframe(
                    q_sources[
                        [
                            "domain",
                            "source_group",
                            "url",
                            "source_type",
                            "used_for",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab5:
        st.subheader("GEO Action Tracker")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("No actions found. Please run: python src/action_log.py --import-csv")
        else:
            total_actions = len(actions_df)
            planned_count = (actions_df["status"] == "planned").sum()
            published_count = (actions_df["status"] == "published").sum()
            in_progress_count = (actions_df["status"] == "in_progress").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Actions", total_actions)
            c2.metric("Planned", planned_count)
            c3.metric("In Progress", in_progress_count)
            c4.metric("Published", published_count)

            st.divider()

            st.markdown("### Action Log")

            st.dataframe(
                actions_df,
                use_container_width=True,
                hide_index=True,
            )

            download_button_for_df(
                "Download Action Log CSV",
                actions_df,
                "geo_action_log.csv",
            )

            st.divider()

            st.markdown("### Priority Actions")

            priority_topics = [
                "delivery",
                "small_space",
                "first_home_buyer",
                "dining_table",
                "ikea_alternative",
                "showroom",
            ]

            for topic in priority_topics:
                topic_actions = actions_df[actions_df["target_topic"] == topic]

                if topic_actions.empty:
                    continue

                for _, row in topic_actions.iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"### {row['action_id']} | {row['action_name']}"
                        )
                        st.write(f"**Status:** {row['status']}")
                        st.write(f"**Target Topic:** {row['target_topic']}")
                        st.write(f"**Action Type:** {row['action_type']}")
                        st.write(f"**Target Source:** {row['target_source']}")
                        st.write(f"**Expected Impact:** {row['expected_impact']}")
                        st.write(f"**Start Date:** {row['start_date']}")
                        st.write(
                            f"**Publish Date:** {row['publish_date'] if row['publish_date'] else 'Not published yet'}"
                        )
                        st.write(f"**Notes:** {row['notes']}")

            st.divider()

            st.markdown("### How to Update Action Status")

            st.code(
                """
# Example: mark action 1 as in progress
python src/action_log.py --action-id 1 --update-status in_progress

# Example: mark action 1 as published
python src/action_log.py --action-id 1 --update-status published --publish-date 2026-05-28
                """.strip(),
                language="bash",
            )

    with tab6:
        st.subheader("Action Impact Comparison")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("No actions found. Please run: python src/action_log.py --import-csv")
        else:
            st.markdown(
                """
                Compare one baseline run against a later run to estimate whether GEO actions improved
                visibility, ranking, source coverage, or risk metrics.
                """
            )

            run_labels = {
                f"Run {row.run_id} | {row.created_at} | {row.notes}": row.run_id
                for row in runs_df.itertuples()
            }

            labels = list(run_labels.keys())

            c1, c2 = st.columns(2)

            with c1:
                before_label = st.selectbox(
                    "Before run",
                    options=labels,
                    index=0 if labels else None,
                    key="before_run_select",
                )

            with c2:
                after_label = st.selectbox(
                    "After run",
                    options=labels,
                    index=0 if labels else None,
                    key="after_run_select",
                )

            before_run_id = run_labels[before_label]
            after_run_id = run_labels[after_label]

            before_answers = load_answers(before_run_id)
            before_sources = load_sources(before_run_id)
            after_answers = load_answers(after_run_id)
            after_sources = load_sources(after_run_id)

            impact_df = build_action_impact(
                actions_df=actions_df,
                before_answers=before_answers,
                before_sources=before_sources,
                after_answers=after_answers,
                after_sources=after_sources,
                before_run_id=before_run_id,
                after_run_id=after_run_id,
            )

            if impact_df.empty:
                st.info("No impact data available.")
            else:
                st.markdown("### Impact Summary")

                total_actions = len(impact_df)
                strong_positive = (impact_df["impact_judgement"] == "strong_positive").sum()
                promising = (impact_df["impact_judgement"] == "positive_or_promising").sum()
                no_change = (impact_df["impact_judgement"] == "no_clear_change").sum()
                negative = (impact_df["impact_judgement"] == "negative").sum()

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Actions Compared", total_actions)
                m2.metric("Strong Positive", strong_positive)
                m3.metric("Promising", promising)
                m4.metric("No Clear Change", no_change)
                m5.metric("Negative", negative)

                st.divider()

                st.markdown("### Action-Level Impact Cards")

                for _, row in impact_df.iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"### {row['action_id']} | {row['action_name']}"
                        )

                        st.write(
                            f"**Topic:** {row['target_topic']} | "
                            f"**Status:** {row['status']} | "
                            f"**Judgement:** {row['impact_judgement']}"
                        )

                        a, b, c, d = st.columns(4)

                        a.metric(
                            "Visibility",
                            f"{row['after_visibility']:.0%}",
                            f"{row['visibility_lift']:+.0%}",
                        )

                        b.metric(
                            "Top-3 Rate",
                            f"{row['after_top3_rate']:.0%}",
                            f"{row['top3_rate_lift']:+.0%}",
                        )

                        c.metric(
                            "Source Coverage",
                            f"{row['after_source_coverage']:.0%}",
                            f"{row['source_coverage_lift']:+.0%}",
                        )

                        if pd.isna(row["after_avg_rank"]):
                            rank_value = "NA"
                        else:
                            rank_value = f"{row['after_avg_rank']:.2f}"

                        if pd.isna(row["avg_rank_change"]):
                            rank_delta = "NA"
                        else:
                            rank_delta = f"{row['avg_rank_change']:+.2f}"

                        d.metric(
                            "Avg Rank",
                            rank_value,
                            rank_delta,
                        )

                        st.write(f"**Expected Impact:** {row['expected_impact']}")
                        st.write(f"**Before Domains:** {row['before_source_domains']}")
                        st.write(f"**After Domains:** {row['after_source_domains']}")

                st.divider()

                st.markdown("### Impact Data Table")

                st.dataframe(
                    impact_df,
                    use_container_width=True,
                    hide_index=True,
                )

                download_button_for_df(
                    "Download Action Impact CSV",
                    impact_df,
                    f"action_impact_before_{before_run_id}_after_{after_run_id}.csv",
                )


if __name__ == "__main__":
    main()