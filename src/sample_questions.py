import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

INPUT_PATH = DATA_DIR / "geo_questions_from_keywords.csv"


REGION_CONFIG = {
    "NZ": {
        "country": "New Zealand",
        "city": "Auckland",
    },
    "AU": {
        "country": "Australia",
        "city": "Sydney",
    },
    "CA": {
        "country": "Canada",
        "city": "Toronto",
    },
}


QUESTION_TEMPLATES = {
    "where_buy_city": "Where can I buy {keyword} in {city}?",
    "best_stores_country": "What are the best stores for {keyword} in {country}?",
    "affordable_country": "Where can I buy affordable {keyword} in {country}?",
    "delivery_country": "Which stores offer reliable delivery for {keyword} in {country}?",
    "reviews_country": "Which stores have good reviews for {keyword} in {country}?",
    "showroom_city": "Which stores have showrooms for {keyword} in {city}?",
    "ikea_alternative_country": "What are the best IKEA alternatives for {keyword} in {country}?",
    "ifurniture_consideration": "Is iFurniture a good place to buy {keyword} in {country}?",
}


INTENT_ORDER = [
    "where_buy_city",
    "best_stores_country",
    "affordable_country",
    "delivery_country",
    "reviews_country",
    "showroom_city",
    "ikea_alternative_country",
    "ifurniture_consideration",
]


CATEGORY_MAP = {
    "outdoor": "outdoor",

    "sofa": "living_room",
    "sofa_bed": "sofa",
    "recliner": "living_room",
    "accent_chair": "living_room",
    "chair": "living_room",

    "bed": "bedroom",
    "bed_frame": "bedroom",
    "bedroom": "bedroom",
    "mattress": "bedroom",
    "bedside_table": "bedroom",
    "wardrobe": "bedroom",
    "dresser": "bedroom",
    "dressing_table": "bedroom",
    "kids_room": "bedroom",

    "dining_table": "dining",
    "dining_set": "dining",
    "dining_chair": "dining",
    "bar_stool": "dining",

    "coffee_table": "living_room",
    "side_table": "living_room",
    "console_table": "living_room",
    "table_general": "general",
    "tv_unit": "living_room",

    "office_chair": "office",
    "desk": "office",
    "office_furniture": "office",

    "storage": "storage",
    "furniture_general": "general",
    "other": "other",
}


def clean_keyword(keyword: str) -> str:
    text = str(keyword).strip().lower()
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    if len(words) % 2 == 0:
        half = len(words) // 2
        if words[:half] == words[half:]:
            text = " ".join(words[:half])

    replacements = {
        "dining table dining table": "dining table",
        "study table study table": "study table",
        "bed double bed": "double bed",
    }

    text = replacements.get(text, text)

    return text


def infer_topic_from_keyword(keyword: str, current_topic: str) -> str:
    kw = clean_keyword(keyword)

    # Specific rules first
    if any(term in kw for term in ["office chair", "desk chair", "ergonomic chair", "computer chair", "office seating"]):
        return "office_chair"

    if any(term in kw for term in ["office furniture", "office cubes"]):
        return "office_furniture"

    if any(term in kw for term in ["sofa bed", "couch bed", "couch with bed", "sleeper sofa", "pull out couch"]):
        return "sofa_bed"

    if any(term in kw for term in ["bed frame", "bed base", "bed bases", "bed and frame", "headboard"]):
        return "bed_frame"

    if any(term in kw for term in ["bedside table", "nightstand", "bed side table", "bedside cabinet", "bedside cupboard", "bed table"]):
        return "bedside_table"

    if any(term in kw for term in ["double bed", "queen bed", "single bed", "king bed", "super king bed", "superking bed", "king single bed", "twin bed", "day bed", "loft bed", "trundle bed", "canopy bed", "bed sets", "bed set", "buy bed", "cheap bed"]):
        return "bed"

    if any(term in kw for term in ["outdoor", "patio", "garden furniture", "deck chair", "sun lounger", "pool lounger", "poolside", "loungers"]):
        return "outdoor"

    if any(term in kw for term in ["recliner"]):
        return "recliner"

    if any(term in kw for term in ["sofa", "couch", "lounge suite"]):
        return "sofa"

    if any(term in kw for term in ["armchair", "accent chair", "occasional chair", "egg chair", "hanging chair", "lounge chair", "chaise"]):
        return "accent_chair"

    if any(term in kw for term in ["dining set", "dining suite", "dining table set", "dinette table set"]):
        return "dining_set"

    if any(term in kw for term in ["dining table", "kitchen table", "dining room table"]):
        return "dining_table"

    if any(term in kw for term in ["dining chair", "dining room chairs"]):
        return "dining_chair"

    if any(term in kw for term in ["coffee table", "cocktail table"]):
        return "coffee_table"

    if any(term in kw for term in ["side table", "side tables", "nest of tables"]):
        return "side_table"

    if any(term in kw for term in ["console table", "hall table", "accent table"]):
        return "console_table"

    if any(term in kw for term in ["dressing table", "vanity table"]):
        return "dressing_table"

    if any(term in kw for term in ["desk", "computer table", "office table", "study table", "standing desk"]):
        return "desk"

    if any(term in kw for term in ["wardrobe", "closet"]):
        return "wardrobe"

    if any(term in kw for term in ["dresser", "tallboy", "chest of drawers", "drawers"]):
        return "dresser"

    if any(term in kw for term in ["storage", "shelf", "shelving", "bookcase", "cabinet"]):
        return "storage"

    if any(term in kw for term in ["tv unit", "tv stand", "tv table", "entertainment unit"]):
        return "tv_unit"

    if any(term in kw for term in ["furniture store", "furniture shops", "furniture warehouse", "living room furniture"]):
        return "furniture_general"

    if any(term in kw for term in ["chair", "chairs"]):
        return "chair"

    if any(term in kw for term in ["table"]):
        return "table_general"

    return current_topic if pd.notna(current_topic) else "other"


def rebuild_question(row: pd.Series) -> str:
    intent = row["intent"]
    keyword = row["keyword"]
    region = row["region"]

    config = REGION_CONFIG.get(region, REGION_CONFIG["NZ"])
    template = QUESTION_TEMPLATES.get(intent)

    if template is None:
        return str(row["question"])

    return template.format(
        keyword=keyword,
        country=config["country"],
        city=config["city"],
    )


def prepare_questions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["keyword"] = df["keyword"].apply(clean_keyword)
    df["topic"] = df.apply(
        lambda row: infer_topic_from_keyword(row["keyword"], row["topic"]),
        axis=1,
    )
    df["category"] = df["topic"].map(CATEGORY_MAP).fillna("other")
    df["question"] = df.apply(rebuild_question, axis=1)

    df["avg_monthly_searches"] = pd.to_numeric(
        df["avg_monthly_searches"],
        errors="coerce",
    ).fillna(0)

    df["geo_priority_score"] = pd.to_numeric(
        df["geo_priority_score"],
        errors="coerce",
    ).fillna(0)

    df = df.drop_duplicates(subset=["region", "question"]).reset_index(drop=True)

    return df


def allocate_even_quotas(items: list[str], capacities: dict[str, int], target_n: int) -> dict[str, int]:
    items = [item for item in items if capacities.get(item, 0) > 0]

    if not items or target_n <= 0:
        return {}

    quota = {item: 0 for item in items}

    base = target_n // len(items)
    remainder = target_n % len(items)

    for i, item in enumerate(items):
        quota[item] = min(base + (1 if i < remainder else 0), capacities[item])

    while sum(quota.values()) < target_n:
        remaining_items = [
            item for item in items
            if quota[item] < capacities[item]
        ]

        if not remaining_items:
            break

        remaining_items = sorted(
            remaining_items,
            key=lambda x: capacities[x] - quota[x],
            reverse=True,
        )

        quota[remaining_items[0]] += 1

    return quota


def allocate_topic_quotas(intent_df: pd.DataFrame, target_n: int) -> dict[str, int]:
    if intent_df.empty or target_n <= 0:
        return {}

    topic_summary = (
        intent_df.groupby("topic")
        .agg(
            available=("question_id", "count"),
            total_searches=("avg_monthly_searches", "sum"),
            avg_priority=("geo_priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["total_searches", "avg_priority"], ascending=[False, False])
    )

    if topic_summary.empty:
        return {}

    total_searches = topic_summary["total_searches"].sum()

    if total_searches <= 0:
        topic_summary["raw_quota"] = target_n / len(topic_summary)
    else:
        topic_summary["raw_quota"] = topic_summary["total_searches"] / total_searches * target_n

    topic_summary["quota"] = topic_summary["raw_quota"].round().astype(int)
    topic_summary.loc[topic_summary["quota"] < 1, "quota"] = 1

    topic_summary["quota"] = topic_summary.apply(
        lambda row: min(row["quota"], row["available"]),
        axis=1,
    )

    while topic_summary["quota"].sum() > target_n:
        candidates = topic_summary[topic_summary["quota"] > 1]

        if candidates.empty:
            break

        idx = candidates.sort_values(
            ["quota", "total_searches"],
            ascending=[False, True],
        ).index[0]

        topic_summary.loc[idx, "quota"] -= 1

    while topic_summary["quota"].sum() < target_n:
        candidates = topic_summary[topic_summary["quota"] < topic_summary["available"]]

        if candidates.empty:
            break

        idx = candidates.sort_values(
            ["total_searches", "avg_priority"],
            ascending=[False, False],
        ).index[0]

        topic_summary.loc[idx, "quota"] += 1

    return dict(zip(topic_summary["topic"], topic_summary["quota"]))


def select_rows(
    candidates: pd.DataFrame,
    n: int,
    used_question_ids: set,
    used_keyword_counts: dict,
    max_per_keyword: int,
) -> list[pd.Series]:
    selected = []

    if candidates.empty or n <= 0:
        return selected

    candidates = candidates[
        ~candidates["question_id"].isin(used_question_ids)
    ].copy()

    candidates = candidates.sort_values(
        ["geo_priority_score", "avg_monthly_searches"],
        ascending=[False, False],
    )

    # Round 1: prefer keywords not over-used
    for _, row in candidates.iterrows():
        keyword_key = (row["region"], row["keyword"])

        if used_keyword_counts[keyword_key] >= max_per_keyword:
            continue

        selected.append(row)
        used_question_ids.add(row["question_id"])
        used_keyword_counts[keyword_key] += 1

        if len(selected) >= n:
            return selected

    # Round 2: relax keyword limit if needed
    for _, row in candidates.iterrows():
        if row["question_id"] in used_question_ids:
            continue

        keyword_key = (row["region"], row["keyword"])

        selected.append(row)
        used_question_ids.add(row["question_id"])
        used_keyword_counts[keyword_key] += 1

        if len(selected) >= n:
            break

    return selected


def sample_region_questions(
    df: pd.DataFrame,
    region: str,
    target_n: int,
    max_per_keyword: int,
) -> pd.DataFrame:
    region_df = df[df["region"] == region].copy()

    if region_df.empty or target_n <= 0:
        return pd.DataFrame(columns=df.columns)

    available_intents = [
        intent for intent in INTENT_ORDER
        if intent in set(region_df["intent"])
    ]

    intent_capacities = {
        intent: len(region_df[region_df["intent"] == intent])
        for intent in available_intents
    }

    intent_quotas = allocate_even_quotas(
        items=available_intents,
        capacities=intent_capacities,
        target_n=target_n,
    )

    selected_rows = []
    used_question_ids = set()
    used_keyword_counts = defaultdict(int)

    for intent in available_intents:
        intent_quota = intent_quotas.get(intent, 0)

        if intent_quota <= 0:
            continue

        intent_df = region_df[region_df["intent"] == intent].copy()

        topic_quotas = allocate_topic_quotas(
            intent_df=intent_df,
            target_n=intent_quota,
        )

        for topic, topic_quota in topic_quotas.items():
            topic_df = intent_df[intent_df["topic"] == topic].copy()

            picked = select_rows(
                candidates=topic_df,
                n=topic_quota,
                used_question_ids=used_question_ids,
                used_keyword_counts=used_keyword_counts,
                max_per_keyword=max_per_keyword,
            )

            selected_rows.extend(picked)

    sampled = pd.DataFrame(selected_rows)

    if sampled.empty:
        sampled = pd.DataFrame(columns=df.columns)

    if len(sampled) < target_n:
        remaining = region_df[
            ~region_df["question_id"].isin(set(sampled["question_id"]) if not sampled.empty else set())
        ].copy()

        needed = target_n - len(sampled)

        picked = select_rows(
            candidates=remaining,
            n=needed,
            used_question_ids=used_question_ids,
            used_keyword_counts=used_keyword_counts,
            max_per_keyword=max_per_keyword,
        )

        if picked:
            sampled = pd.concat([sampled, pd.DataFrame(picked)], ignore_index=True)

    if len(sampled) > target_n:
        sampled = sampled.sort_values(
            ["geo_priority_score", "avg_monthly_searches"],
            ascending=[False, False],
        ).head(target_n)

    return sampled.reset_index(drop=True)


def build_sample(
    df: pd.DataFrame,
    nz: int,
    au: int,
    ca: int,
    max_per_keyword: int,
) -> pd.DataFrame:
    parts = []

    for region, target_n in [("NZ", nz), ("AU", au), ("CA", ca)]:
        part = sample_region_questions(
            df=df,
            region=region,
            target_n=target_n,
            max_per_keyword=max_per_keyword,
        )
        parts.append(part)

    sampled = pd.concat(parts, ignore_index=True)

    sampled = sampled.drop_duplicates(subset=["region", "question"]).reset_index(drop=True)

    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled["question_id"] = [f"SQ{i:04d}" for i in range(1, len(sampled) + 1)]

    return sampled


def print_summary(sampled: pd.DataFrame) -> None:
    print("=" * 90)
    print("STRATIFIED GEO QUESTION SAMPLE")
    print("=" * 90)

    print("\nQuestions by region:")
    print(sampled.groupby("region").size())

    print("\nQuestions by region and intent:")
    print(sampled.groupby(["region", "intent"]).size())

    print("\nQuestions by topic:")
    topic_summary = (
        sampled.groupby(["region", "topic"])
        .agg(
            question_count=("question", "count"),
            total_searches=("avg_monthly_searches", "sum"),
            avg_priority=("geo_priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["region", "question_count"], ascending=[True, False])
    )

    print(topic_summary.to_string(index=False))

    print("\nTop 40 sampled questions:")
    cols = [
        "question_id",
        "region",
        "topic",
        "intent",
        "keyword",
        "avg_monthly_searches",
        "question",
    ]
    print(sampled[cols].head(40).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified sample from keyword-generated GEO questions."
    )

    parser.add_argument("--nz", type=int, default=180)
    parser.add_argument("--au", type=int, default=60)
    parser.add_argument("--ca", type=int, default=60)
    parser.add_argument("--max-per-keyword", type=int, default=2)
    parser.add_argument("--write-questions", action="store_true")

    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. Please run generate_questions_from_keywords.py first."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "question_id",
        "region",
        "topic",
        "category",
        "intent",
        "keyword",
        "question",
        "avg_monthly_searches",
        "geo_priority_score",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Input file missing columns: {missing}")

    df = prepare_questions(df)

    sampled = build_sample(
        df=df,
        nz=args.nz,
        au=args.au,
        ca=args.ca,
        max_per_keyword=args.max_per_keyword,
    )

    sample_size = len(sampled)

    sample_full_path = DATA_DIR / f"questions_sample_{sample_size}_full.csv"
    sample_run_path = DATA_DIR / f"questions_sample_{sample_size}.csv"
    summary_path = OUTPUTS_DIR / f"questions_sample_{sample_size}_summary.csv"

    sampled.to_csv(sample_full_path, index=False)

    run_ready_cols = ["question_id", "region", "topic", "category", "question"]
    sampled[run_ready_cols].to_csv(sample_run_path, index=False)

    summary = (
        sampled.groupby(["region", "topic", "category"])
        .agg(
            question_count=("question", "count"),
            total_searches=("avg_monthly_searches", "sum"),
            avg_priority=("geo_priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["region", "question_count"], ascending=[True, False])
    )

    summary.to_csv(summary_path, index=False)

    print_summary(sampled)

    print("\nSaved files:")
    print(f"- {sample_full_path}")
    print(f"- {sample_run_path}")
    print(f"- {summary_path}")

    if args.write_questions:
        final_questions_path = DATA_DIR / "questions.csv"
        sampled[run_ready_cols].to_csv(final_questions_path, index=False)
        print(f"\nWrote sampled run-ready questions to: {final_questions_path}")
    else:
        print("\nPreview mode only. data/questions.csv was NOT overwritten.")
        print("To overwrite data/questions.csv, rerun with --write-questions")


if __name__ == "__main__":
    main()