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
    "harveynorman.co.nz",
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


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_domain(domain: str) -> str:
    if not domain:
        return ""
    return str(domain).strip().lower().replace("www.", "")


def classify_source(domain: str) -> str:
    domain = normalize_domain(domain)

    if domain in {"ifurniture.co.nz"}:
        return "自有信源"

    if domain in COMMUNITY_DOMAINS:
        return "社区论坛"

    if domain in MARKETPLACE_DOMAINS:
        return "交易平台"

    if domain in MEDIA_GUIDE_DOMAINS:
        return "媒体/城市指南"

    if domain in AUTHORITY_DOMAINS:
        return "公益/权威机构"

    if domain in COMPETITOR_DOMAINS:
        return "竞品/零售商"

    if domain.endswith(".co.nz") or domain.endswith(".nz"):
        return "新西兰本地信源"

    return "其他信源"


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
        return pd.DataFrame(columns=["品牌", "出现次数"])

    return (
        pd.DataFrame(
            [{"品牌": brand, "出现次数": count} for brand, count in counts.items()]
        )
        .sort_values("出现次数", ascending=False)
        .reset_index(drop=True)
    )


def build_risk_count(answers_df: pd.DataFrame) -> pd.DataFrame:
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


def build_domain_summary(sources_df: pd.DataFrame) -> pd.DataFrame:
    if sources_df.empty:
        return pd.DataFrame(
            columns=[
                "信源域名",
                "信源类型",
                "引用次数",
                "覆盖问题数",
                "覆盖话题",
                "问题编号",
                "示例链接",
            ]
        )

    rows = []

    for domain, g in sources_df.groupby("domain"):
        rows.append(
            {
                "信源域名": domain,
                "信源类型": classify_source(domain),
                "引用次数": len(g),
                "覆盖问题数": g["question_id"].nunique(),
                "覆盖话题": ", ".join(sorted(g["topic"].dropna().astype(str).unique())),
                "问题编号": ", ".join(
                    sorted(g["question_id"].dropna().astype(str).unique())
                ),
                "示例链接": g["url"].dropna().iloc[0]
                if len(g["url"].dropna())
                else "",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["覆盖问题数", "引用次数", "信源域名"],
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
            if classify_source(domain) == "竞品/零售商"
        ]

        rows.append(
            {
                "问题编号": answer["question_id"],
                "地区": answer["region"],
                "话题": answer["topic"],
                "品类": answer["category"],
                "问题": answer["question"],
                "是否提到 iFurniture": bool(answer["ifurniture_mentioned"]),
                "iFurniture 排名": answer["ifurniture_rank"],
                "情感": translate_sentiment(answer["ifurniture_sentiment"]),
                "是否提到风险": bool(answer["risk_mentioned"]),
                "信源数量": len(q_sources),
                "独立域名数量": len(domains),
                "是否引用 ifurniture 官网": has_ifurniture_source,
                "引用域名": ", ".join(domains),
                "信源类型": ", ".join(source_groups),
                "竞品信源": ", ".join(competitor_domains),
            }
        )

    return pd.DataFrame(rows)


def recommend_action(row: pd.Series) -> str:
    topic = str(row.get("话题", "")).lower()
    question = str(row.get("问题", "")).lower()

    if "delivery" in topic or "delivery" in question:
        return "创建或优化物流透明度与可靠配送页面"

    if "small" in topic or "sofa" in question or "living room" in question:
        return "创建小户型沙发 / 小客厅沙发购买指南"

    if "first_home" in topic or "first-home" in question or "first home" in question:
        return "创建首次置业者家具购买指南"

    if "dining" in topic or "dining table" in question:
        return "创建新西兰高性价比餐桌购买指南"

    if "ikea" in topic or "ikea" in question:
        return "创建 IKEA 替代品牌对比页面"

    if "showroom" in topic or "showroom" in question:
        return "优化 Onehunga 展厅页面，增加结构化信息"

    if "rental" in topic or "airbnb" in question or "property" in question:
        return "强化出租房 / Airbnb / 房产投资家具套餐页面"

    return "创建对应话题的 GEO 落地页，并争取第三方信源引用"


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
                rank_int = int(rank)

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
                    "问题编号": row["问题编号"],
                    "话题": row["话题"],
                    "问题": row["问题"],
                    "是否提到 iFurniture": row["是否提到 iFurniture"],
                    "iFurniture 排名": row["iFurniture 排名"],
                    "是否引用 ifurniture 官网": row["是否引用 ifurniture 官网"],
                    "优先级分数": priority_score,
                    "机会原因": " | ".join(reasons),
                    "建议行动": recommend_action(row),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "问题编号",
                "话题",
                "问题",
                "是否提到 iFurniture",
                "iFurniture 排名",
                "是否引用 ifurniture 官网",
                "优先级分数",
                "机会原因",
                "建议行动",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["优先级分数", "问题编号"], ascending=[False, True])
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
                "行动ID": action["action_id"],
                "行动名称": action["action_name"],
                "状态": translate_status(action["status"]),
                "目标话题": topic,
                "目标信源": action["target_source"],
                "预期影响": action["expected_impact"],
                "发布日期": action["publish_date"],
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


def download_button_for_df(label: str, df: pd.DataFrame, file_name: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def main():
    st.title("📊 iFurniture GEO 表现追踪器")
    st.caption(
        "MVP 仪表盘：追踪 AI 曝光、推荐排名、信源引用、优先优化机会、行动记录和行动效果。"
    )

    if not Path(DB_PATH).exists():
        st.error("未找到数据库。请确认 data/geo_tracker.db 已上传到 GitHub。")
        return

    runs_df = load_runs()

    if runs_df.empty:
        st.warning("暂无运行记录。请先运行 python src/run_batch.py")
        return

    st.sidebar.header("运行批次选择")

    run_options = {
        f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}": row.run_id
        for row in runs_df.itertuples()
    }

    labels = list(run_options.keys())

    default_index = 0
    for i, row in enumerate(runs_df.itertuples()):
        if int(row.run_id) == 6:
            default_index = i
            break

    selected_label = st.sidebar.selectbox(
        "选择一个运行批次",
        options=labels,
        index=default_index,
    )

    selected_run_id = run_options[selected_label]
    selected_run = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]

    answers_df = load_answers(selected_run_id)
    sources_df = load_sources(selected_run_id)

    st.sidebar.markdown("### 当前批次信息")
    st.sidebar.write(f"**Run ID：** {selected_run['run_id']}")
    st.sidebar.write(f"**样本数：** {selected_run['answer_count']}")
    st.sidebar.write(f"**模型：** {selected_run['model']}")
    st.sidebar.write(f"**地区：** {selected_run['region']}")
    st.sidebar.write(f"**创建时间：** {selected_run['created_at']}")
    st.sidebar.write(f"**备注：** {selected_run['notes']}")


    if answers_df.empty:
        st.warning("当前批次没有回答数据。")
        return

    metrics = calculate_metrics(answers_df, sources_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("问题/样本数", metrics["total"])
    col2.metric("AI 曝光率", f"{metrics['visibility']:.0%}")
    col3.metric("首位推荐率", f"{metrics['first_rate']:.0%}")
    col4.metric("Top-3 推荐率", f"{metrics['top3_rate']:.0%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("风险提及率", f"{metrics['risk_rate']:.0%}")
    col6.metric("官网信源覆盖率", f"{metrics['ifurniture_source_coverage']:.0%}")
    col7.metric("信源引用次数", metrics["total_sources"])
    col8.metric("独立信源域名数", metrics["unique_domains"])

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "总览",
            "信源分析",
            "优先机会",
            "原始回答",
            "行动追踪",
            "行动效果",
        ]
    )

    with tab1:
        st.subheader("问题级 GEO 结果")

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

        display_df = display_df.rename(
            columns={
                "question_id": "问题编号",
                "region": "地区",
                "topic": "话题",
                "category": "品类",
                "ifurniture_mentioned": "是否提到 iFurniture",
                "ifurniture_rank": "iFurniture 排名",
                "ifurniture_sentiment": "情感",
                "risk_mentioned": "是否提到风险",
                "question": "问题",
            }
        )

        display_df["是否提到 iFurniture"] = display_df["是否提到 iFurniture"].astype(bool)
        display_df["是否提到风险"] = display_df["是否提到风险"].astype(bool)
        display_df["情感"] = display_df["情感"].apply(translate_sentiment)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("高频出现品牌")
            brand_df = build_brand_count(answers_df)

            if brand_df.empty:
                st.info("暂无品牌统计数据。")
            else:
                st.bar_chart(brand_df.set_index("品牌")["出现次数"])
                st.dataframe(brand_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("风险短语")
            risk_df = build_risk_count(answers_df)

            if risk_df.empty:
                st.success("当前批次未发现风险短语。")
            else:
                st.bar_chart(risk_df.set_index("风险短语")["出现次数"])
                st.dataframe(risk_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("高频信源域名")

        domain_summary_df = build_domain_summary(sources_df)

        if domain_summary_df.empty:
            st.info("暂无信源引用。请确认该批次使用了 web_search=True。")
        else:
            chart_df = domain_summary_df.head(15).set_index("信源域名")["覆盖问题数"]
            st.bar_chart(chart_df)

            st.dataframe(domain_summary_df, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载信源域名统计 CSV",
                domain_summary_df,
                f"source_domain_summary_run_{selected_run_id}.csv",
            )

        st.divider()

        st.subheader("问题级信源覆盖")

        question_source_df = build_question_source_summary(answers_df, sources_df)

        if question_source_df.empty:
            st.info("暂无问题级信源统计。")
        else:
            st.dataframe(question_source_df, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载问题级信源统计 CSV",
                question_source_df,
                f"question_source_summary_run_{selected_run_id}.csv",
            )

    with tab3:
        st.subheader("优先 GEO 优化机会")

        question_source_df = build_question_source_summary(answers_df, sources_df)
        opportunity_df = build_opportunity_summary(question_source_df)

        if opportunity_df.empty:
            st.success("当前批次暂无高优先级优化机会。")
        else:
            top_priority = opportunity_df.head(5)

            for _, row in top_priority.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"### {row['问题编号']} | {row['话题']} | 优先级分数：{row['优先级分数']}"
                    )
                    st.write(f"**问题：** {row['问题']}")
                    st.write(f"**建议行动：** {row['建议行动']}")
                    st.write(f"**机会原因：** {row['机会原因']}")

            st.divider()

            st.dataframe(opportunity_df, use_container_width=True, hide_index=True)

            download_button_for_df(
                "下载优先机会 CSV",
                opportunity_df,
                f"source_opportunities_run_{selected_run_id}.csv",
            )

    with tab4:
        st.subheader("原始 AI 回答")

        question_labels = {
            f"{row.question_id} | {row.topic} | rank={row.ifurniture_rank} | {row.question[:70]}": row.answer_id
            for row in answers_df.itertuples()
        }

        selected_question_label = st.selectbox(
            "选择一条回答查看",
            options=list(question_labels.keys()),
        )

        selected_answer_id = question_labels[selected_question_label]
        selected_answer = answers_df[
            answers_df["answer_id"] == selected_answer_id
        ].iloc[0]

        st.markdown("### 问题")
        st.write(selected_answer["question"])

        st.markdown("### AI 原始回答")
        st.write(selected_answer["raw_answer"])

        st.markdown("### 解析结果")

        extracted = {
            "是否提到 iFurniture": bool(selected_answer["ifurniture_mentioned"]),
            "iFurniture 排名": selected_answer["ifurniture_rank"],
            "情感": translate_sentiment(selected_answer["ifurniture_sentiment"]),
            "是否提到风险": bool(selected_answer["risk_mentioned"]),
            "风险短语": safe_json_loads(selected_answer["risk_phrases_json"], []),
            "提到的品牌": safe_json_loads(selected_answer["brands_json"], []),
            "信源": safe_json_loads(selected_answer["sources_json"], []),
        }

        st.json(extracted)

        st.markdown("### 该回答引用的信源")

        if sources_df.empty:
            st.info("当前批次没有信源引用。")
        else:
            q_sources = sources_df[sources_df["answer_id"] == selected_answer_id]

            if q_sources.empty:
                st.info("该回答没有信源引用。")
            else:
                source_display = q_sources[
                    [
                        "domain",
                        "source_group",
                        "url",
                        "source_type",
                        "used_for",
                    ]
                ].copy()

                source_display = source_display.rename(
                    columns={
                        "domain": "信源域名",
                        "source_group": "信源类型",
                        "url": "链接",
                        "source_type": "引用类型",
                        "used_for": "引用标题/用途",
                    }
                )

                st.dataframe(source_display, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("GEO 行动追踪")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("暂无行动记录。请先运行：python src/action_log.py --import-csv")
        else:
            actions_display_df = actions_df.copy()
            actions_display_df["status_cn"] = actions_display_df["status"].apply(translate_status)

            total_actions = len(actions_display_df)
            planned_count = (actions_display_df["status"] == "planned").sum()
            published_count = (actions_display_df["status"] == "published").sum()
            in_progress_count = (actions_display_df["status"] == "in_progress").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("行动总数", total_actions)
            c2.metric("计划中", planned_count)
            c3.metric("进行中", in_progress_count)
            c4.metric("已发布", published_count)

            st.divider()

            st.markdown("### 行动台账")

            action_table = actions_display_df[
                [
                    "action_id",
                    "action_name",
                    "action_type",
                    "target_region",
                    "target_topic",
                    "target_source",
                    "expected_impact",
                    "status_cn",
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
                    "status_cn": "状态",
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

            st.markdown("### 优先行动")

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
                        st.markdown(f"### {row['action_id']} | {row['action_name']}")
                        st.write(f"**状态：** {translate_status(row['status'])}")
                        st.write(f"**目标话题：** {row['target_topic']}")
                        st.write(f"**行动类型：** {row['action_type']}")
                        st.write(f"**目标信源：** {row['target_source']}")
                        st.write(f"**预期影响：** {row['expected_impact']}")
                        st.write(f"**开始日期：** {row['start_date']}")
                        st.write(
                            f"**发布日期：** {row['publish_date'] if row['publish_date'] else '尚未发布'}"
                        )
                        st.write(f"**备注：** {row['notes']}")

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

    with tab6:
        st.subheader("行动效果对比")

        actions_df = load_actions()

        if actions_df.empty:
            st.warning("暂无行动记录。请先运行：python src/action_log.py --import-csv")
        else:
            st.markdown(
                """
                选择一个行动前的基准 run 和一个行动后的 run，比较 GEO 指标是否出现提升。
                当前如果选择同一个 run，结果应为「暂无明显变化」。
                """
            )

            run_labels = {
                f"Run {row.run_id} | 样本数 {row.answer_count} | {row.created_at}": row.run_id
                for row in runs_df.itertuples()
            }

            labels = list(run_labels.keys())

            default_before_index = 0
            for i, row in enumerate(runs_df.itertuples()):
                if int(row.run_id) == 6:
                    default_before_index = i
                    break

            c1, c2 = st.columns(2)

            with c1:
                before_label = st.selectbox(
                    "Before run（行动前）",
                    options=labels,
                    index=default_before_index,
                    key="before_run_select",
                )

            with c2:
                after_label = st.selectbox(
                    "After run（行动后）",
                    options=labels,
                    index=default_before_index,
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
                st.info("暂无可对比的行动效果数据。")
            else:
                st.markdown("### 效果概览")

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

                        if pd.isna(row["After 平均排名"]):
                            rank_value = "NA"
                        else:
                            rank_value = f"{row['After 平均排名']:.2f}"

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
                    f"action_impact_before_{before_run_id}_after_{after_run_id}.csv",
                )


if __name__ == "__main__":
    main()