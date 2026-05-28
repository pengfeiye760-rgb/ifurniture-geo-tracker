import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import DB_PATH


OWNED_DOMAINS = {
    "ifurniture.co.nz",
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
    "harveynorman.co.nz",
    "farmers.co.nz",
    "mitre10.co.nz",
    "bunnings.co.nz",
    "treasurebox.co.nz",
    "pbtech.co.nz",
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
    page_title="iFurniture GEO 表现追踪器",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_domain(domain: str) -> str:
    if not domain:
        return ""
    return str(domain).strip().lower().replace("www.", "")


def domain_in_set(domain: str, domain_set: set[str]) -> bool:
    domain = normalize_domain(domain)

    for target in domain_set:
        target = normalize_domain(target)
        if domain == target or domain.endswith("." + target):
            return True

    return False


def classify_source(domain: str) -> str:
    domain = normalize_domain(domain)

    if domain_in_set(domain, OWNED_DOMAINS):
        return "owned_ifurniture"

    if domain_in_set(domain, COMPETITOR_DOMAINS):
        return "competitor_or_retailer"

    if domain_in_set(domain, MARKETPLACE_DOMAINS):
        return "marketplace"

    if domain_in_set(domain, COMMUNITY_DOMAINS):
        return "community"

    if domain_in_set(domain, MEDIA_GUIDE_DOMAINS):
        return "media_or_city_guide"

    if domain_in_set(domain, AUTHORITY_DOMAINS):
        return "authority_or_nonprofit"

    if domain.endswith(".co.nz") or domain.endswith(".nz"):
        return "nz_local_source"

    return "other"


def translate_source_group(value: str) -> str:
    mapping = {
        "owned_ifurniture": "iFurniture 自有信源",
        "competitor_or_retailer": "竞品/零售商信源",
        "marketplace": "交易平台",
        "community": "社区论坛",
        "media_or_city_guide": "媒体/城市指南",
        "authority_or_nonprofit": "权威/公益机构",
        "nz_local_source": "新西兰本地信源",
        "other": "其他信源",
    }
    return mapping.get(str(value), str(value))


def translate_sentiment(value: str) -> str:
    mapping = {
        "positive": "正面",
        "neutral": "中性",
        "negative": "负面",
        "not_mentioned": "未提及",
    }
    return mapping.get(str(value), str(value))


def translate_status(value: str) -> str:
    mapping = {
        "planned": "计划中",
        "in_progress": "进行中",
        "published": "已发布",
        "paused": "暂停",
        "cancelled": "取消",
    }
    return mapping.get(str(value), str(value))


def translate_impact(value: str) -> str:
    mapping = {
        "strong_positive": "显著正向",
        "positive_or_promising": "有改善迹象",
        "no_clear_change": "暂无明显变化",
        "mixed_or_uncertain": "结果不确定",
        "negative": "负向变化",
    }
    return mapping.get(str(value), str(value))


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


def format_rank(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def download_button_for_df(label: str, df: pd.DataFrame, file_name: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

def load_runs() -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            r.run_id,
            r.run_name,
            r.model,
            r.region,
            r.notes,
            r.created_at,
            COUNT(a.answer_id) AS answer_count
        FROM runs r
        LEFT JOIN answers a
            ON r.run_id = a.run_id
        GROUP BY
            r.run_id,
            r.run_name,
            r.model,
            r.region,
            r.notes,
            r.created_at
        ORDER BY r.run_id DESC;
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
            COALESCE(s.region, a.region) AS region,
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
    df["source_group_cn"] = df["source_group"].apply(translate_source_group)

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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def filter_by_region(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    region_filter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if region_filter == "All":
        return answers_df.copy(), sources_df.copy()

    filtered_answers = answers_df[answers_df["region"] == region_filter].copy()

    if sources_df.empty:
        filtered_sources = sources_df.copy()
    else:
        filtered_sources = sources_df[sources_df["region"] == region_filter].copy()

    return filtered_answers, filtered_sources


def calculate_metrics(answers_df: pd.DataFrame, sources_df: pd.DataFrame) -> dict:
    total = len(answers_df)

    if total == 0:
        return {
            "total": 0,
            "visibility": 0.0,
            "first_rate": 0.0,
            "top3_rate": 0.0,
            "risk_rate": 0.0,
            "avg_rank": None,
            "ifurniture_source_coverage": 0.0,
            "total_sources": 0,
            "unique_domains": 0,
        }

    mentioned = answers_df["ifurniture_mentioned"].fillna(0).astype(int).sum()

    ranks = pd.to_numeric(answers_df["ifurniture_rank"], errors="coerce")
    first = (ranks == 1).sum()
    top3 = ranks.apply(lambda x: pd.notna(x) and int(x) <= 3).sum()

    risk = answers_df["risk_mentioned"].fillna(0).astype(int).sum()
    rank_average = avg_rank(answers_df["ifurniture_rank"])

    if sources_df.empty:
        ifurniture_source_coverage = 0.0
        total_sources = 0
        unique_domains = 0
    else:
        answer_has_ifurniture_source = (
            sources_df[sources_df["source_group"] == "owned_ifurniture"]["answer_id"]
            .dropna()
            .unique()
        )
        ifurniture_source_coverage = safe_rate(len(answer_has_ifurniture_source), total)
        total_sources = len(sources_df)
        unique_domains = sources_df["domain"].nunique()

    return {
        "total": total,
        "visibility": safe_rate(mentioned, total),
        "first_rate": safe_rate(first, total),
        "top3_rate": safe_rate(top3, total),
        "risk_rate": safe_rate(risk, total),
        "avg_rank": rank_average,
        "ifurniture_source_coverage": ifurniture_source_coverage,
        "total_sources": total_sources,
        "unique_domains": unique_domains,
    }


def build_region_summary(answers_df: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if answers_df.empty:
        return pd.DataFrame()

    for region in sorted(answers_df["region"].dropna().astype(str).unique()):
        region_answers, region_sources = filter_by_region(answers_df, sources_df, region)
        metrics = calculate_metrics(region_answers, region_sources)

        rows.append(
            {
                "地区": region,
                "样本数": metrics["total"],
                "AI 曝光率": metrics["visibility"],
                "首位推荐率": metrics["first_rate"],
                "Top-3 推荐率": metrics["top3_rate"],
                "平均排名": metrics["avg_rank"],
                "官网信源覆盖率": metrics["ifurniture_source_coverage"],
                "风险提及率": metrics["risk_rate"],
                "信源引用次数": metrics["total_sources"],
                "独立信源域名数": metrics["unique_domains"],
            }
        )

    return pd.DataFrame(rows)


def build_topic_region_summary(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if answers_df.empty:
        return pd.DataFrame()

    group_cols = ["region", "topic", "category"]

    for (region, topic, category), group in answers_df.groupby(group_cols, dropna=False):
        answer_ids = set(group["answer_id"].tolist())

        if sources_df.empty:
            topic_sources = pd.DataFrame()
        else:
            topic_sources = sources_df[sources_df["answer_id"].isin(answer_ids)].copy()

        metrics = calculate_metrics(group, topic_sources)

        if topic_sources.empty:
            competitor_domains = ""
            owned_domains = ""
            top_domains = ""
        else:
            competitor_domains = ", ".join(
                sorted(
                    topic_sources[topic_sources["source_group"] == "competitor_or_retailer"]["domain"]
                    .dropna()
                    .astype(str)
                    .unique()
                )[:8]
            )

            owned_domains = ", ".join(
                sorted(
                    topic_sources[topic_sources["source_group"] == "owned_ifurniture"]["domain"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )

            top_domain_counts = (
                topic_sources.groupby("domain")
                .size()
                .sort_values(ascending=False)
                .head(5)
            )

            top_domains = ", ".join(
                [f"{domain}({count})" for domain, count in top_domain_counts.items()]
            )

        rows.append(
            {
                "地区": region,
                "Topic": topic,
                "品类": category,
                "样本数": metrics["total"],
                "AI 曝光率": metrics["visibility"],
                "首位推荐率": metrics["first_rate"],
                "Top-3 推荐率": metrics["top3_rate"],
                "平均排名": metrics["avg_rank"],
                "官网信源覆盖率": metrics["ifurniture_source_coverage"],
                "风险提及率": metrics["risk_rate"],
                "信源引用次数": metrics["total_sources"],
                "独立信源域名数": metrics["unique_domains"],
                "iFurniture 信源": owned_domains,
                "主要竞品信源": competitor_domains,
                "高频信源": top_domains,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(
        ["地区", "样本数", "AI 曝光率", "官网信源覆盖率"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_brand_count(answers_df: pd.DataFrame) -> pd.DataFrame:
    if answers_df.empty or "brands_json" not in answers_df.columns:
        return pd.DataFrame(columns=["品牌", "出现次数"])

    counts = {}

    for value in answers_df["brands_json"].fillna("[]"):
        brands = safe_json_loads(value, [])
        for brand in brands:
            counts[brand] = counts.get(brand, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["品牌", "出现次数"])

    return (
        pd.DataFrame(
            [{"品牌": brand, "出现次数": count} for brand, count in counts.items()]
        )
        .sort_values("出现次数", ascending=False)
        .reset_index(drop=True)
    )


def build_risk_count(answers_df: pd.DataFrame) -> pd.DataFrame:
    if answers_df.empty or "risk_phrases_json" not in answers_df.columns:
        return pd.DataFrame(columns=["风险短语", "出现次数"])

    counts = {}

    for value in answers_df["risk_phrases_json"].fillna("[]"):
        phrases = safe_json_loads(value, [])
        for phrase in phrases:
            counts[phrase] = counts.get(phrase, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["风险短语", "出现次数"])

    return (
        pd.DataFrame(
            [{"风险短语": phrase, "出现次数": count} for phrase, count in counts.items()]
        )
        .sort_values("出现次数", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Source intelligence
# ---------------------------------------------------------------------

def build_domain_summary(sources_df: pd.DataFrame) -> pd.DataFrame:
    if sources_df.empty:
        return pd.DataFrame(
            columns=[
                "地区",
                "信源域名",
                "信源类型",
                "引用次数",
                "覆盖样本数",
                "覆盖话题",
                "示例链接",
            ]
        )

    rows = []

    for (region, domain), g in sources_df.groupby(["region", "domain"], dropna=False):
        rows.append(
            {
                "地区": region,
                "信源域名": domain,
                "信源类型": translate_source_group(classify_source(domain)),
                "引用次数": len(g),
                "覆盖样本数": g["answer_id"].nunique(),
                "覆盖话题": ", ".join(sorted(g["topic"].dropna().astype(str).unique())),
                "示例链接": g["url"].dropna().iloc[0]
                if len(g["url"].dropna())
                else "",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["地区", "覆盖样本数", "引用次数"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def build_source_group_summary(sources_df: pd.DataFrame) -> pd.DataFrame:
    if sources_df.empty:
        return pd.DataFrame(
            columns=[
                "地区",
                "信源类型",
                "引用次数",
                "覆盖样本数",
                "独立域名数",
            ]
        )

    rows = []

    for (region, source_group), g in sources_df.groupby(["region", "source_group"], dropna=False):
        rows.append(
            {
                "地区": region,
                "信源类型": translate_source_group(source_group),
                "引用次数": len(g),
                "覆盖样本数": g["answer_id"].nunique(),
                "独立域名数": g["domain"].nunique(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["地区", "引用次数"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_question_source_summary(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, answer in answers_df.iterrows():
        answer_id = answer["answer_id"]

        if sources_df.empty:
            q_sources = pd.DataFrame()
        else:
            q_sources = sources_df[sources_df["answer_id"] == answer_id].copy()

        if q_sources.empty:
            domains = []
            source_groups = []
            competitor_domains = []
            has_ifurniture_source = False
        else:
            domains = sorted(q_sources["domain"].dropna().astype(str).unique())
            source_groups = sorted(q_sources["source_group_cn"].dropna().astype(str).unique())
            competitor_domains = sorted(
                q_sources[q_sources["source_group"] == "competitor_or_retailer"]["domain"]
                .dropna()
                .astype(str)
                .unique()
            )
            has_ifurniture_source = (
                q_sources["source_group"].eq("owned_ifurniture").any()
            )

        rows.append(
            {
                "answer_id": answer_id,
                "问题编号": answer["question_id"],
                "地区": answer["region"],
                "Topic": answer["topic"],
                "品类": answer["category"],
                "问题": answer["question"],
                "是否提到 iFurniture": bool(answer["ifurniture_mentioned"]),
                "iFurniture 排名": answer["ifurniture_rank"],
                "情感": translate_sentiment(answer["ifurniture_sentiment"]),
                "是否提到风险": bool(answer["risk_mentioned"]),
                "信源数量": len(q_sources),
                "是否引用 ifurniture 官网": has_ifurniture_source,
                "引用域名": ", ".join(domains),
                "信源类型": ", ".join(source_groups),
                "竞品信源": ", ".join(competitor_domains),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------

def recommend_action(row: pd.Series) -> str:
    topic = str(row.get("Topic", "")).lower()
    question = str(row.get("问题", "")).lower()

    if "delivery" in question:
        return "创建或优化配送可靠性 / 物流透明度页面"

    if "showroom" in question:
        return "优化展厅页面，增加结构化门店与产品体验信息"

    if "review" in question:
        return "补充第三方评价信源，并优化评价页 / 产品评价结构"

    if "ikea" in question:
        return "创建 IKEA 替代品牌 / 产品对比页面"

    if topic in {"sofa", "sofa_bed", "recliner", "accent_chair"}:
        return "创建沙发 / 沙发床 / 客厅家具购买指南"

    if topic in {"bed", "bed_frame", "bedroom", "mattress", "bedside_table", "wardrobe", "dresser"}:
        return "创建卧室家具购买指南和对应品类页面"

    if topic in {"dining_table", "dining_set", "dining_chair", "bar_stool"}:
        return "创建餐厅家具购买指南和对应品类页面"

    if topic in {"desk", "office_chair", "office_furniture"}:
        return "创建办公家具购买指南和办公家具 GEO 页面"

    if topic == "outdoor":
        return "创建户外家具购买指南和配送/耐用性说明页面"

    if topic in {"tv_unit", "coffee_table", "side_table", "console_table"}:
        return "创建客厅收纳 / 桌几类产品购买指南"

    return "创建对应话题 GEO 落地页，并争取第三方高信任信源引用"


def build_opportunity_summary(question_df: pd.DataFrame) -> pd.DataFrame:
    if question_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in question_df.iterrows():
        reasons = []
        priority_score = 0

        if not row["是否提到 iFurniture"]:
            reasons.append("AI 回答中未提到 iFurniture")
            priority_score += 5

        rank = row["iFurniture 排名"]

        if pd.notna(rank):
            try:
                rank_int = int(float(rank))

                if rank_int > 3:
                    reasons.append(f"iFurniture 排名未进入前三，目前排名：{rank_int}")
                    priority_score += 3
                elif rank_int == 1:
                    priority_score -= 2

            except Exception:
                pass

        if not row["是否引用 ifurniture 官网"]:
            reasons.append("AI 未引用 ifurniture.co.nz 作为信源")
            priority_score += 4

        if row["竞品信源"]:
            reasons.append(f"存在竞品信源：{row['竞品信源']}")
            priority_score += 1

        if row["是否提到风险"]:
            reasons.append("回答中出现风险或负面短语")
            priority_score += 3

        if priority_score > 0:
            rows.append(
                {
                    "地区": row["地区"],
                    "问题编号": row["问题编号"],
                    "Topic": row["Topic"],
                    "品类": row["品类"],
                    "问题": row["问题"],
                    "是否提到 iFurniture": row["是否提到 iFurniture"],
                    "iFurniture 排名": row["iFurniture 排名"],
                    "是否引用 ifurniture 官网": row["是否引用 ifurniture 官网"],
                    "优先级分数": priority_score,
                    "机会原因": " | ".join(reasons),
                    "建议行动": recommend_action(row),
                    "竞品信源": row["竞品信源"],
                    "引用域名": row["引用域名"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "地区",
                "问题编号",
                "Topic",
                "品类",
                "问题",
                "是否提到 iFurniture",
                "iFurniture 排名",
                "是否引用 ifurniture 官网",
                "优先级分数",
                "机会原因",
                "建议行动",
                "竞品信源",
                "引用域名",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["地区", "优先级分数", "问题编号"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_opportunity_region_summary(opportunity_df: pd.DataFrame) -> pd.DataFrame:
    if opportunity_df.empty:
        return pd.DataFrame(
            columns=["地区", "机会数量", "平均优先级分数", "最高优先级分数"]
        )

    return (
        opportunity_df.groupby("地区")
        .agg(
            机会数量=("问题编号", "count"),
            平均优先级分数=("优先级分数", "mean"),
            最高优先级分数=("优先级分数", "max"),
        )
        .reset_index()
        .sort_values("机会数量", ascending=False)
    )


def build_opportunity_topic_summary(opportunity_df: pd.DataFrame) -> pd.DataFrame:
    if opportunity_df.empty:
        return pd.DataFrame(
            columns=["地区", "Topic", "机会数量", "平均优先级分数", "最高优先级分数"]
        )

    return (
        opportunity_df.groupby(["地区", "Topic"])
        .agg(
            机会数量=("问题编号", "count"),
            平均优先级分数=("优先级分数", "mean"),
            最高优先级分数=("优先级分数", "max"),
        )
        .reset_index()
        .sort_values(["地区", "机会数量", "最高优先级分数"], ascending=[True, False, False])
    )


# ---------------------------------------------------------------------
# Action impact
# ---------------------------------------------------------------------

def topic_metrics(
    answers_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    topic: str,
    region: str = "All",
) -> dict:
    topic_answers = answers_df[answers_df["topic"] == topic].copy()

    if region != "All":
        topic_answers = topic_answers[topic_answers["region"] == region].copy()

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
            "source_domains": "",
        }

    if sources_df.empty:
        topic_sources = pd.DataFrame()
    else:
        answer_ids = set(topic_answers["answer_id"].tolist())
        topic_sources = sources_df[sources_df["answer_id"].isin(answer_ids)].copy()

    metrics = calculate_metrics(topic_answers, topic_sources)

    if topic_sources.empty:
        ifurniture_source_citations = 0
        source_domains = ""
    else:
        ifurniture_source_citations = len(
            topic_sources[topic_sources["source_group"] == "owned_ifurniture"]
        )
        source_domains = ", ".join(
            sorted(topic_sources["domain"].dropna().astype(str).unique())
        )

    return {
        "sample_count": n,
        "visibility_rate": metrics["visibility"],
        "first_rate": metrics["first_rate"],
        "top3_rate": metrics["top3_rate"],
        "risk_rate": metrics["risk_rate"],
        "avg_rank": metrics["avg_rank"],
        "ifurniture_source_coverage": metrics["ifurniture_source_coverage"],
        "ifurniture_source_citations": ifurniture_source_citations,
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
        "source_coverage_lift": after["ifurniture_source_coverage"] - before["ifurniture_source_coverage"],
        "source_citation_change": after["ifurniture_source_citations"] - before["ifurniture_source_citations"],
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
    region: str = "All",
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
            region=region,
        )

        after = topic_metrics(
            answers_df=after_answers,
            sources_df=after_sources,
            topic=topic,
            region=region,
        )

        lift = calculate_lift(before, after)
        impact_judgement = judge_impact(lift)

        rows.append(
            {
                "行动ID": action["action_id"],
                "行动名称": action["action_name"],
                "状态": translate_status(action["status"]),
                "目标地区": action["target_region"],
                "目标话题": topic,
                "目标信源": action["target_source"],
                "预期影响": action["expected_impact"],
                "发布日期": action["publish_date"],
                "对比区域": region,
                "Before Run": before_run_id,
                "After Run": after_run_id,
                "Before 样本数": before["sample_count"],
                "After 样本数": after["sample_count"],
                "Before 曝光率": before["visibility_rate"],
                "After 曝光率": after["visibility_rate"],
                "曝光率变化": lift["visibility_lift"],
                "Before 首位推荐率": before["first_rate"],
                "After 首位推荐率": after["first_rate"],
                "首位推荐率变化": lift["first_rate_lift"],
                "Before Top3率": before["top3_rate"],
                "After Top3率": after["top3_rate"],
                "Top3率变化": lift["top3_rate_lift"],
                "Before 风险率": before["risk_rate"],
                "After 风险率": after["risk_rate"],
                "风险率变化": lift["risk_rate_change"],
                "Before 平均排名": before["avg_rank"],
                "After 平均排名": after["avg_rank"],
                "平均排名变化": lift["avg_rank_change"],
                "Before 官网信源覆盖率": before["ifurniture_source_coverage"],
                "After 官网信源覆盖率": after["ifurniture_source_coverage"],
                "官网信源覆盖率变化": lift["source_coverage_lift"],
                "Before 官网引用次数": before["ifurniture_source_citations"],
                "After 官网引用次数": after["ifurniture_source_citations"],
                "官网引用次数变化": lift["source_citation_change"],
                "Before 信源域名": before["source_domains"],
                "After 信源域名": after["source_domains"],
                "效果判断": translate_impact(impact_judgement),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def show_metric_cards(metrics: dict, label_prefix: str = "") -> None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(f"{label_prefix}样本数", metrics["total"])
    col2.metric(f"{label_prefix}AI 曝光率", f"{metrics['visibility']:.0%}")
    col3.metric(f"{label_prefix}首位推荐率", f"{metrics['first_rate']:.0%}")
    col4.metric(f"{label_prefix}Top-3 推荐率", f"{metrics['top3_rate']:.0%}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(f"{label_prefix}平均排名", format_rank(metrics["avg_rank"]))
    col6.metric(f"{label_prefix}官网信源覆盖率", f"{metrics['ifurniture_source_coverage']:.0%}")
    col7.metric(f"{label_prefix}风险提及率", f"{metrics['risk_rate']:.0%}")
    col8.metric(f"{label_prefix}独立信源域名数", metrics["unique_domains"])


def format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()

    for col in columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.0%}"
            )

    return formatted


def format_rank_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()

    for col in columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(format_rank)

    return formatted


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    st.title("📊 iFurniture GEO 表现追踪器")
    st.caption(
        "区域化 MVP 仪表盘：追踪 NZ / AU / CA 的 AI 曝光、推荐排名、官网信源覆盖、竞品信源和 GEO 优化机会。"
    )

    if not Path(DB_PATH).exists():
        st.error("未找到数据库。请确认 data/geo_tracker.db 已存在。")
        return

    runs_df = load_runs()

    if runs_df.empty:
        st.warning("暂无运行记录。请先运行 python src/run_batch.py")
        return

    st.sidebar.header("运行批次选择")

    run_labels = [
        f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}"
        for row in runs_df.itertuples()
    ]

    run_id_by_label = {
        f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}": row.run_id
        for row in runs_df.itertuples()
    }

    default_run_id = (
        runs_df.sort_values(["answer_count", "run_id"], ascending=[False, False])
        .iloc[0]["run_id"]
    )
    default_index = list(runs_df["run_id"]).index(default_run_id)

    selected_run_label = st.sidebar.selectbox(
        "选择一个运行批次",
        options=run_labels,
        index=default_index,
    )

    selected_run_id = run_id_by_label[selected_run_label]
    selected_run = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]

    answers_df = load_answers(selected_run_id)
    sources_df = load_sources(selected_run_id)

    if answers_df.empty:
        st.warning("当前批次没有回答数据。")
        return

    available_regions = sorted(answers_df["region"].dropna().astype(str).unique())
    region_options = ["All"] + available_regions

    region_filter = st.sidebar.selectbox(
        "地区过滤",
        options=region_options,
        index=0,
        help="选择 All 查看整体，或选择 NZ / AU / CA 查看单一区域。",
    )

    filtered_answers, filtered_sources = filter_by_region(
        answers_df=answers_df,
        sources_df=sources_df,
        region_filter=region_filter,
    )

    st.sidebar.markdown("### 当前批次信息")
    st.sidebar.write(f"**Run ID：** {selected_run['run_id']}")
    st.sidebar.write(f"**样本数：** {selected_run['answer_count']}")
    st.sidebar.write(f"**模型：** {selected_run['model']}")
    st.sidebar.write(f"**地区：** {selected_run['region']}")
    st.sidebar.write(f"**创建时间：** {selected_run['created_at']}")
    st.sidebar.write(f"**备注：** {selected_run['notes']}")
    st.sidebar.write(f"**当前过滤：** {region_filter}")

    st.divider()

    metrics = calculate_metrics(filtered_answers, filtered_sources)

    if region_filter == "All":
        st.subheader("总体 GEO 表现")
    else:
        st.subheader(f"{region_filter} 区域 GEO 表现")

    show_metric_cards(metrics)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "区域表现",
            "Topic 矩阵",
            "信源分析",
            "优先机会",
            "原始回答",
            "行动追踪",
            "行动效果",
        ]
    )

    # -----------------------------------------------------------------
    # Tab 1: Region performance
    # -----------------------------------------------------------------
    with tab1:
        st.subheader("NZ / AU / CA 区域表现对比")

        region_summary_df = build_region_summary(answers_df, sources_df)

        if region_summary_df.empty:
            st.info("暂无区域表现数据。")
        else:
            display_df = format_percent_columns(
                region_summary_df,
                [
                    "AI 曝光率",
                    "首位推荐率",
                    "Top-3 推荐率",
                    "官网信源覆盖率",
                    "风险提及率",
                ],
            )
            display_df = format_rank_columns(display_df, ["平均排名"])

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            chart_df = region_summary_df.set_index("地区")[
                ["AI 曝光率", "Top-3 推荐率", "官网信源覆盖率"]
            ]

            st.markdown("### 区域核心指标图")
            st.bar_chart(chart_df)

            download_button_for_df(
                "下载区域表现 CSV",
                region_summary_df,
                f"region_performance_run_{selected_run_id}.csv",
            )

        st.divider()

        st.subheader("当前过滤区域的问题级结果")

        question_display_df = filtered_answers[
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

        question_display_df = question_display_df.rename(
            columns={
                "question_id": "问题编号",
                "region": "地区",
                "topic": "Topic",
                "category": "品类",
                "ifurniture_mentioned": "是否提到 iFurniture",
                "ifurniture_rank": "iFurniture 排名",
                "ifurniture_sentiment": "情感",
                "risk_mentioned": "是否提到风险",
                "question": "问题",
            }
        )

        question_display_df["是否提到 iFurniture"] = question_display_df["是否提到 iFurniture"].astype(bool)
        question_display_df["是否提到风险"] = question_display_df["是否提到风险"].astype(bool)
        question_display_df["情感"] = question_display_df["情感"].apply(translate_sentiment)

        st.dataframe(question_display_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # Tab 2: Topic matrix
    # -----------------------------------------------------------------
    with tab2:
        st.subheader("Region × Topic 表现矩阵")

        topic_region_df = build_topic_region_summary(filtered_answers, filtered_sources)

        if topic_region_df.empty:
            st.info("暂无 Topic 矩阵数据。")
        else:
            display_df = format_percent_columns(
                topic_region_df,
                [
                    "AI 曝光率",
                    "首位推荐率",
                    "Top-3 推荐率",
                    "官网信源覆盖率",
                    "风险提及率",
                ],
            )
            display_df = format_rank_columns(display_df, ["平均排名"])

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("### Topic 曝光率图")

            chart_source = topic_region_df.copy()
            chart_source["Region-Topic"] = chart_source["地区"].astype(str) + " | " + chart_source["Topic"].astype(str)

            st.bar_chart(
                chart_source.set_index("Region-Topic")[["AI 曝光率", "官网信源覆盖率", "Top-3 推荐率"]]
            )

            download_button_for_df(
                "下载 Topic 矩阵 CSV",
                topic_region_df,
                f"topic_region_matrix_run_{selected_run_id}_{region_filter}.csv",
            )

    # -----------------------------------------------------------------
    # Tab 3: Source intelligence
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("信源类型结构")

        source_group_df = build_source_group_summary(filtered_sources)

        if source_group_df.empty:
            st.info("当前过滤条件下暂无信源引用。")
        else:
            st.dataframe(source_group_df, use_container_width=True, hide_index=True)

            chart_df = source_group_df.copy()
            chart_df["地区-信源类型"] = chart_df["地区"].astype(str) + " | " + chart_df["信源类型"].astype(str)

            st.bar_chart(chart_df.set_index("地区-信源类型")["引用次数"])

        st.divider()

        st.subheader("高频信源域名")

        domain_summary_df = build_domain_summary(filtered_sources)

        if domain_summary_df.empty:
            st.info("当前过滤条件下暂无信源域名。")
        else:
            st.dataframe(domain_summary_df, use_container_width=True, hide_index=True)

            top_domain_chart = domain_summary_df.head(20).copy()
            top_domain_chart["地区-域名"] = (
                top_domain_chart["地区"].astype(str) + " | " + top_domain_chart["信源域名"].astype(str)
            )

            st.bar_chart(top_domain_chart.set_index("地区-域名")["覆盖样本数"])

            download_button_for_df(
                "下载信源域名统计 CSV",
                domain_summary_df,
                f"source_domain_summary_run_{selected_run_id}_{region_filter}.csv",
            )

        st.divider()

        st.subheader("问题级信源覆盖")

        question_source_df = build_question_source_summary(filtered_answers, filtered_sources)

        if question_source_df.empty:
            st.info("暂无问题级信源数据。")
        else:
            st.dataframe(question_source_df, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载问题级信源统计 CSV",
                question_source_df,
                f"question_source_summary_run_{selected_run_id}_{region_filter}.csv",
            )

    # -----------------------------------------------------------------
    # Tab 4: Opportunities
    # -----------------------------------------------------------------
    with tab4:
        st.subheader("优先 GEO 优化机会")

        question_source_df = build_question_source_summary(filtered_answers, filtered_sources)
        opportunity_df = build_opportunity_summary(question_source_df)

        if opportunity_df.empty:
            st.success("当前过滤条件下暂无高优先级优化机会。")
        else:
            st.markdown("### 区域机会概览")

            opportunity_region_df = build_opportunity_region_summary(opportunity_df)
            st.dataframe(opportunity_region_df, use_container_width=True, hide_index=True)

            st.markdown("### Topic 机会概览")

            opportunity_topic_df = build_opportunity_topic_summary(opportunity_df)
            st.dataframe(opportunity_topic_df, use_container_width=True, hide_index=True)

            st.divider()

            st.markdown("### Top 优先机会卡片")

            for _, row in opportunity_df.head(12).iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"### {row['地区']} | {row['问题编号']} | {row['Topic']} | 优先级：{row['优先级分数']}"
                    )
                    st.write(f"**问题：** {row['问题']}")
                    st.write(f"**建议行动：** {row['建议行动']}")
                    st.write(f"**机会原因：** {row['机会原因']}")
                    st.write(f"**竞品信源：** {row['竞品信源']}")
                    st.write(f"**引用域名：** {row['引用域名']}")

            st.divider()

            st.markdown("### 完整机会表")
            st.dataframe(opportunity_df, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载优先机会 CSV",
                opportunity_df,
                f"geo_opportunities_run_{selected_run_id}_{region_filter}.csv",
            )

    # -----------------------------------------------------------------
    # Tab 5: Raw answers
    # -----------------------------------------------------------------
    with tab5:
        st.subheader("原始 AI 回答")

        if filtered_answers.empty:
            st.info("当前过滤条件下暂无回答。")
        else:
            question_labels = {
                f"{row.question_id} | {row.region} | {row.topic} | rank={row.ifurniture_rank} | {row.question[:80]}": row.answer_id
                for row in filtered_answers.itertuples()
            }

            selected_question_label = st.selectbox(
                "选择一条回答查看",
                options=list(question_labels.keys()),
            )

            selected_answer_id = question_labels[selected_question_label]
            selected_answer = filtered_answers[
                filtered_answers["answer_id"] == selected_answer_id
            ].iloc[0]

            st.markdown("### 问题")
            st.write(selected_answer["question"])

            st.markdown("### AI 原始回答")
            st.write(selected_answer["raw_answer"])

            st.markdown("### 解析结果")

            extracted = {
                "地区": selected_answer["region"],
                "Topic": selected_answer["topic"],
                "是否提到 iFurniture": bool(selected_answer["ifurniture_mentioned"]),
                "iFurniture 排名": selected_answer["ifurniture_rank"],
                "情感": translate_sentiment(selected_answer["ifurniture_sentiment"]),
                "是否提到风险": bool(selected_answer["risk_mentioned"]),
                "风险短语": safe_json_loads(selected_answer.get("risk_phrases_json", "[]"), []),
                "提到的品牌": safe_json_loads(selected_answer.get("brands_json", "[]"), []),
                "信源": safe_json_loads(selected_answer.get("sources_json", "[]"), []),
            }

            st.json(extracted)

            st.markdown("### 该回答引用的信源")

            if filtered_sources.empty:
                st.info("当前过滤条件下没有信源引用。")
            else:
                q_sources = filtered_sources[filtered_sources["answer_id"] == selected_answer_id]

                if q_sources.empty:
                    st.info("该回答没有信源引用。")
                else:
                    source_display = q_sources[
                        [
                            "domain",
                            "source_group_cn",
                            "url",
                            "source_type",
                            "used_for",
                        ]
                    ].copy()

                    source_display = source_display.rename(
                        columns={
                            "domain": "信源域名",
                            "source_group_cn": "信源类型",
                            "url": "链接",
                            "source_type": "引用类型",
                            "used_for": "引用标题/用途",
                        }
                    )

                    st.dataframe(source_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # Tab 6: Action tracker
    # -----------------------------------------------------------------
    with tab6:
        st.subheader("GEO 行动追踪")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("暂无行动记录。请先运行：python src/action_log.py --import-csv")
        else:
            actions_display_df = actions_df.copy()
            actions_display_df["状态"] = actions_display_df["status"].apply(translate_status)

            total_actions = len(actions_display_df)
            planned_count = (actions_display_df["status"] == "planned").sum()
            in_progress_count = (actions_display_df["status"] == "in_progress").sum()
            published_count = (actions_display_df["status"] == "published").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("行动总数", total_actions)
            c2.metric("计划中", planned_count)
            c3.metric("进行中", in_progress_count)
            c4.metric("已发布", published_count)

            st.divider()

            action_table = actions_display_df[
                [
                    "action_id",
                    "action_name",
                    "action_type",
                    "target_region",
                    "target_topic",
                    "target_source",
                    "expected_impact",
                    "状态",
                    "start_date",
                    "publish_date",
                    "notes",
                ]
            ].copy()

            action_table = action_table.rename(
                columns={
                    "action_id": "行动ID",
                    "action_name": "行动名称",
                    "action_type": "行动类型",
                    "target_region": "目标地区",
                    "target_topic": "目标话题",
                    "target_source": "目标信源",
                    "expected_impact": "预期影响",
                    "start_date": "开始日期",
                    "publish_date": "发布日期",
                    "notes": "备注",
                }
            )

            st.dataframe(action_table, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载行动台账 CSV",
                action_table,
                "geo_action_log.csv",
            )

            st.divider()

            st.markdown("### 如何更新行动状态")

            st.code(
                """
# 示例：将行动 1 标记为进行中
python src/action_log.py --action-id 1 --update-status in_progress

# 示例：将行动 1 标记为已发布
python src/action_log.py --action-id 1 --update-status published --publish-date 2026-05-28
                """.strip(),
                language="bash",
            )

    # -----------------------------------------------------------------
    # Tab 7: Action impact
    # -----------------------------------------------------------------
    with tab7:
        st.subheader("行动效果对比")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("暂无行动记录。请先运行：python src/action_log.py --import-csv")
        else:
            st.markdown(
                """
                选择一个行动前的基准 run 和一个行动后的 run，比较不同 GEO 行动对应 topic 的变化。
                可选择 All / NZ / AU / CA 进行区域化对比。
                """
            )

            impact_region = st.selectbox(
                "效果对比区域",
                options=region_options,
                index=0,
                key="impact_region_select",
            )

            all_run_labels = [
                f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}"
                for row in runs_df.itertuples()
            ]

            all_run_id_by_label = {
                f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}": row.run_id
                for row in runs_df.itertuples()
            }

            c1, c2 = st.columns(2)

            with c1:
                before_label = st.selectbox(
                    "Before run（行动前）",
                    options=all_run_labels,
                    index=default_index,
                    key="before_run_select",
                )

            with c2:
                after_label = st.selectbox(
                    "After run（行动后）",
                    options=all_run_labels,
                    index=default_index,
                    key="after_run_select",
                )

            before_run_id = all_run_id_by_label[before_label]
            after_run_id = all_run_id_by_label[after_label]

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
                region=impact_region,
            )

            if impact_df.empty:
                st.info("暂无可对比的行动效果数据。")
            else:
                total_actions = len(impact_df)
                strong_positive = (impact_df["效果判断"] == "显著正向").sum()
                promising = (impact_df["效果判断"] == "有改善迹象").sum()
                no_change = (impact_df["效果判断"] == "暂无明显变化").sum()
                negative = (impact_df["效果判断"] == "负向变化").sum()

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("对比行动数", total_actions)
                m2.metric("显著正向", strong_positive)
                m3.metric("有改善迹象", promising)
                m4.metric("暂无明显变化", no_change)
                m5.metric("负向变化", negative)

                st.divider()

                st.markdown("### 行动级效果卡片")

                for _, row in impact_df.iterrows():
                    with st.container(border=True):
                        st.markdown(f"### {row['行动ID']} | {row['行动名称']}")

                        st.write(
                            f"**目标话题：** {row['目标话题']} | "
                            f"**状态：** {row['状态']} | "
                            f"**对比区域：** {row['对比区域']} | "
                            f"**效果判断：** {row['效果判断']}"
                        )

                        a, b, c, d = st.columns(4)

                        a.metric(
                            "曝光率",
                            f"{row['After 曝光率']:.0%}",
                            f"{row['曝光率变化']:+.0%}",
                        )

                        b.metric(
                            "Top-3 率",
                            f"{row['After Top3率']:.0%}",
                            f"{row['Top3率变化']:+.0%}",
                        )

                        c.metric(
                            "官网信源覆盖率",
                            f"{row['After 官网信源覆盖率']:.0%}",
                            f"{row['官网信源覆盖率变化']:+.0%}",
                        )

                        rank_value = format_rank(row["After 平均排名"])

                        if pd.isna(row["平均排名变化"]):
                            rank_delta = "NA"
                        else:
                            rank_delta = f"{row['平均排名变化']:+.2f}"

                        d.metric(
                            "平均排名",
                            rank_value,
                            rank_delta,
                            help="排名数字越小越好，例如 1 优于 3。",
                        )

                        st.write(f"**预期影响：** {row['预期影响']}")
                        st.write(f"**Before 信源：** {row['Before 信源域名']}")
                        st.write(f"**After 信源：** {row['After 信源域名']}")

                st.divider()

                st.markdown("### 效果对比数据表")

                st.dataframe(impact_df, use_container_width=True, hide_index=True)

                download_button_for_df(
                    "下载行动效果对比 CSV",
                    impact_df,
                    f"action_impact_before_{before_run_id}_after_{after_run_id}_{impact_region}.csv",
                )


if __name__ == "__main__":
    main()