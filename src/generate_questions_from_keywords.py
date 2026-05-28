import argparse
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

DEFAULT_FILES = {
    "NZ": DATA_DIR / "keyword_nz.csv",
    "AU": DATA_DIR / "keyword_au.csv",
    "CA": DATA_DIR / "keyword_ca.csv",
}


PRODUCT_TOPIC_RULES = [
    # More specific rules first
    ("sofa_bed", ["sofa bed", "couch bed", "pull out couch", "sleeper sofa"]),
    ("bedside_table", ["bedside table", "nightstand", "bed side table", "bedside cabinet", "bedside stand", "bed table", "bedside cupboard"]),
    ("bed_frame", ["bed frame", "bed base", "queen bed frame", "double bed frame", "king bed frame", "bed and frame", "queen bed and frame"]),

    # Beds
    ("bed", [
        "double bed", "queen bed", "single bed", "king bed", "super king bed",
        "superking bed", "king single bed", "twin bed", "day bed", "loft bed",
        "trundle bed", "canopy bed", "cheap bed", "buy bed", "bed sets",
        "bed set", "bed for sale"
    ]),
    ("bedroom", ["bedroom furniture", "bedroom set", "bedroom suite", "bedroom"]),
    ("mattress", ["mattress"]),

    # Outdoor
    ("outdoor", [
        "outdoor furniture", "patio", "garden furniture", "outdoor table",
        "outdoor chair", "outdoor sofa", "outdoor seating", "outdoor settee",
        "outdoor setting", "outdoor living", "deck chairs", "sun lounger",
        "poolside chairs", "outdoor lounge", "outdoor side table",
        "outdoor dining", "table and chairs for outside"
    ]),

    # Sofas and chairs
    ("recliner", ["recliner", "recliners", "recliner chair"]),
    ("sofa", ["sofa", "couch", "lounge suite", "lounge furniture", "corner couch", "sectional sofa"]),
    ("accent_chair", [
        "armchair", "armchairs", "accent chair", "accent chairs",
        "occasional chair", "occasional chairs", "egg chair", "egg chairs",
        "hanging chair", "lounge chair", "chaise lounge", "chaise loungers",
        "lounger chaise"
    ]),
    ("chair", ["chair", "chairs"]),

    # Tables
    ("dining_set", ["dining set", "dining suite", "dining table set", "dinette table set"]),
    ("dining_table", ["dining table", "kitchen table", "dining room table", "wood dining room table"]),
    ("dining_chair", ["dining chair", "dining room chairs"]),
    ("bar_stool", ["bar stool", "counter stool"]),
    ("coffee_table", ["coffee table", "cocktail table", "round cocktail table", "glass cocktail table"]),
    ("side_table", ["side table", "side tables", "round side tables", "nest of tables"]),
    ("console_table", ["console table", "hall table", "hall tables", "accent table", "round accent table"]),
    ("dressing_table", ["dressing table", "vanity table", "mirrored dressing table"]),
    ("table_general", ["round tables", "trestle table", "table"]),

    # Office
    ("office_chair", ["office chair", "desk chair", "ergonomic chair", "computer chair", "office seating", "chair office ergonomic", "rolling chair", "rolly chair"]),
    ("desk", ["desk", "standing desk", "computer desk", "office desk", "study desk", "computer table", "office table", "study table"]),
    ("office_furniture", ["office furniture", "office cubes"]),

    # Storage / cabinets
    ("wardrobe", ["wardrobe", "closet", "clothes rack"]),
    ("dresser", ["dresser", "tallboy", "chest of drawers", "drawers"]),
    ("storage", ["storage cabinet", "shelf", "shelving", "bookcase", "cabinet", "storage ottoman", "storage ottomans"]),
    ("tv_unit", ["tv unit", "tv stand", "entertainment unit", "tv table"]),

    # Kids
    ("kids_room", ["kids furniture", "bunk bed", "kids bed", "children furniture", "kids single bed", "childrens single bed"]),

    # General
    ("furniture_general", ["furniture", "furniture store", "furniture shops", "furniture warehouse"]),
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


REGION_CONFIG = {
    "NZ": {
        "country": "New Zealand",
        "city": "Auckland",
        "local_brand": "iFurniture",
        "competitor": "IKEA",
    },
    "AU": {
        "country": "Australia",
        "city": "Sydney",
        "local_brand": "iFurniture",
        "competitor": "IKEA",
    },
    "CA": {
        "country": "Canada",
        "city": "Toronto",
        "local_brand": "iFurniture",
        "competitor": "IKEA",
    },
}


QUESTION_TEMPLATES = [
    ("where_buy_city", "Where can I buy {keyword} in {city}?"),
    ("best_stores_country", "What are the best stores for {keyword} in {country}?"),
    ("affordable_country", "Where can I buy affordable {keyword} in {country}?"),
    ("delivery_country", "Which stores offer reliable delivery for {keyword} in {country}?"),
    ("reviews_country", "Which stores have good reviews for {keyword} in {country}?"),
    ("showroom_city", "Which stores have showrooms for {keyword} in {city}?"),
    ("ikea_alternative_country", "What are the best IKEA alternatives for {keyword} in {country}?"),
    ("ifurniture_consideration", "Is iFurniture a good place to buy {keyword} in {country}?"),
]


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:4]

    if raw.startswith(b"\xff\xfe"):
        return "utf-16"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    return "utf-8"


def detect_header_and_sep(path: Path, encoding: str) -> tuple[int, str]:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = [f.readline() for _ in range(30)]

    header_idx = 0
    header_line = lines[0] if lines else ""

    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "keyword" in line_lower and (
            "avg. monthly searches" in line_lower
            or "monthly searches" in line_lower
            or "competition" in line_lower
        ):
            header_idx = i
            header_line = line
            break

    pipe_count = header_line.count("|")
    tab_count = header_line.count("\t")
    comma_count = header_line.count(",")

    if tab_count >= pipe_count and tab_count >= comma_count and tab_count > 0:
        sep = "\t"
    elif pipe_count >= comma_count and pipe_count > 0:
        sep = "|"
    else:
        sep = ","

    return header_idx, sep


def read_keyword_csv(path: Path) -> pd.DataFrame:
    encodings_to_try = []

    detected = detect_encoding(path)
    encodings_to_try.append(detected)

    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin1"]:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)

    last_error = None

    for encoding in encodings_to_try:
        try:
            header_idx, sep = detect_header_and_sep(path, encoding)

            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=sep,
                skiprows=header_idx,
                engine="python",
            )

            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

            if len(df.columns) >= 2 and any("keyword" in str(c).lower() for c in df.columns):
                print(f"  Read OK | encoding={encoding} | sep={repr(sep)} | skiprows={header_idx}")
                print(f"  Columns: {list(df.columns)[:8]}")
                return df

            last_error = ValueError(
                f"File read but keyword column not detected. columns={list(df.columns)}"
            )

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not read keyword CSV file: {path}. Last error: {repr(last_error)}"
    )


def normalize_col_name(col: str) -> str:
    return str(col).strip().lower().replace("\ufeff", "")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_col_name(c): c for c in df.columns}

    for candidate in candidates:
        candidate_norm = normalize_col_name(candidate)
        if candidate_norm in normalized:
            return normalized[candidate_norm]

    for col in df.columns:
        col_norm = normalize_col_name(col)
        for candidate in candidates:
            if normalize_col_name(candidate) in col_norm:
                return col

    return None


def clean_number(value) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).strip().lower()

    if text in {"", "nan", "none", "--", "-"}:
        return 0.0

    multiplier = 1.0

    if "k" in text:
        multiplier = 1000.0
        text = text.replace("k", "")

    if "m" in text:
        multiplier = 1000000.0
        text = text.replace("m", "")

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("usd", "")
    text = text.replace("nzd", "")
    text = text.replace("aud", "")
    text = text.replace("cad", "")
    text = text.replace("%", "")
    text = re.sub(r"[^0-9.\-]", "", text)

    if text in {"", "-", "."}:
        return 0.0

    try:
        return float(text) * multiplier
    except Exception:
        return 0.0


def competition_to_score(value) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).strip().lower()

    if text in {"high", "h"}:
        return 1.0
    if text in {"medium", "med", "m"}:
        return 0.6
    if text in {"low", "l"}:
        return 0.3

    num = clean_number(value)

    if num > 1:
        return min(num / 100, 1.0)

    return max(min(num, 1.0), 0.0)


def infer_topic(keyword: str) -> str:
    kw = str(keyword).lower()

    for topic, patterns in PRODUCT_TOPIC_RULES:
        for pattern in patterns:
            if pattern in kw:
                return topic

    return "other"


def is_relevant_keyword(keyword: str) -> bool:
    kw = str(keyword).lower()

    relevant_terms = [
        "furniture",
        "sofa",
        "couch",
        "chair",
        "table",
        "desk",
        "bed",
        "mattress",
        "wardrobe",
        "dresser",
        "drawer",
        "cabinet",
        "shelf",
        "stool",
        "tv unit",
        "tv stand",
        "outdoor",
        "patio",
        "dining",
        "office",
        "bedside",
        "nightstand",
        "coffee table",
        "lounge",
        "recliner",
        "storage",
        "bookcase",
        "bunk",
    ]

    return any(term in kw for term in relevant_terms)


def load_keyword_file(region: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found for {region}: {path}")

    df = read_keyword_csv(path)

    keyword_col = find_column(
        df,
        ["keyword", "search term", "search terms", "keywords"],
    )

    volume_col = find_column(
        df,
        ["avg. monthly searches", "avg monthly searches", "volume", "search volume", "monthly searches"],
    )

    competition_col = find_column(
        df,
        ["competition", "competitive density", "competition index", "competition (indexed value)"],
    )

    bid_high_col = find_column(
        df,
        ["top of page bid (high range)", "top page bid high", "high bid", "cpc", "top of page bid high"],
    )

    bid_low_col = find_column(
        df,
        ["top of page bid (low range)", "top page bid low", "low bid", "top of page bid low"],
    )

    if keyword_col is None:
        raise ValueError(
            f"Cannot find keyword column in {path}. Columns found: {list(df.columns)}"
        )

    # 关键修复：必须用 df.index 创建同长度 DataFrame，否则 region 会变成 NaN
    result = pd.DataFrame(index=df.index)

    result["region"] = region
    result["keyword"] = df[keyword_col].astype(str).str.strip()

    if volume_col:
        result["avg_monthly_searches"] = df[volume_col].apply(clean_number)
    else:
        result["avg_monthly_searches"] = 0.0

    if competition_col:
        result["competition_raw"] = df[competition_col].astype(str)
        result["competition_score"] = df[competition_col].apply(competition_to_score)
    else:
        result["competition_raw"] = ""
        result["competition_score"] = 0.0

    if bid_high_col:
        result["top_page_bid_high"] = df[bid_high_col].apply(clean_number)
    else:
        result["top_page_bid_high"] = 0.0

    if bid_low_col:
        result["top_page_bid_low"] = df[bid_low_col].apply(clean_number)
    else:
        result["top_page_bid_low"] = 0.0

    result = result[result["keyword"].str.len() > 1].copy()
    result = result.drop_duplicates(subset=["region", "keyword"]).reset_index(drop=True)

    result["topic"] = result["keyword"].apply(infer_topic)
    result["category"] = result["topic"].map(CATEGORY_MAP).fillna("other")
    result["is_relevant"] = result["keyword"].apply(is_relevant_keyword)

    print(
        f"  Loaded {region}: rows={len(result)}, "
        f"relevant={int(result['is_relevant'].sum())}, "
        f"volume_sum={int(result['avg_monthly_searches'].sum())}"
    )

    return result


def add_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Keyword dataframe is empty before scoring.")

    scored = []

    for region, g in df.groupby("region", dropna=False):
        g = g.copy()

        if pd.isna(region) or str(region).strip() == "":
            continue

        max_volume = max(float(g["avg_monthly_searches"].max()), 1.0)
        max_bid = max(float(g["top_page_bid_high"].max()), 1.0)

        g["volume_score"] = g["avg_monthly_searches"] / max_volume
        g["bid_score"] = g["top_page_bid_high"] / max_bid

        g["geo_priority_score"] = (
            g["volume_score"] * 60
            + g["bid_score"] * 20
            + g["competition_score"] * 10
            + g["is_relevant"].astype(int) * 10
        )

        g = g.sort_values("avg_monthly_searches", ascending=False).reset_index(drop=True)
        g["volume_rank"] = range(1, len(g) + 1)

        total_volume = max(float(g["avg_monthly_searches"].sum()), 1.0)
        g["cumulative_volume"] = g["avg_monthly_searches"].cumsum()
        g["cumulative_volume_share"] = g["cumulative_volume"] / total_volume

        row_cutoff = max(int(len(g) * 0.2), 1)
        g["is_top_20pct_by_row"] = g["volume_rank"] <= row_cutoff
        g["is_top_80pct_volume_cover"] = g["cumulative_volume_share"] <= 0.8

        scored.append(g)

    if not scored:
        raise ValueError("No valid regional keyword groups found after scoring.")

    return pd.concat(scored, ignore_index=True)


def select_keywords(
    df: pd.DataFrame,
    max_nz: int,
    max_au: int,
    max_ca: int,
    only_relevant: bool = True,
) -> pd.DataFrame:
    caps = {
        "NZ": max_nz,
        "AU": max_au,
        "CA": max_ca,
    }

    selected = []

    for region, cap in caps.items():
        g = df[df["region"] == region].copy()

        if only_relevant:
            g = g[g["is_relevant"]].copy()

        g = g[g["is_top_20pct_by_row"]].copy()
        g = g.sort_values("geo_priority_score", ascending=False).head(cap)

        if not g.empty:
            selected.append(g)

    if not selected:
        raise ValueError("No keywords selected. Check relevance rules or input CSV files.")

    return pd.concat(selected, ignore_index=True)


def generate_questions(
    selected_keywords: pd.DataFrame,
    max_templates_per_keyword: int,
) -> pd.DataFrame:
    rows = []
    counter = 1

    templates = QUESTION_TEMPLATES[:max_templates_per_keyword]

    for _, row in selected_keywords.iterrows():
        region = row["region"]
        config = REGION_CONFIG[region]

        keyword = str(row["keyword"]).strip()
        topic = row["topic"]
        category = row["category"]

        for intent, template in templates:
            question = template.format(
                keyword=keyword,
                country=config["country"],
                city=config["city"],
                local_brand=config["local_brand"],
                competitor=config["competitor"],
            )

            rows.append(
                {
                    "question_id": f"KQ{counter:05d}",
                    "region": region,
                    "topic": topic,
                    "category": category,
                    "intent": intent,
                    "keyword": keyword,
                    "question": question,
                    "avg_monthly_searches": row["avg_monthly_searches"],
                    "competition_raw": row["competition_raw"],
                    "competition_score": row["competition_score"],
                    "top_page_bid_high": row["top_page_bid_high"],
                    "top_page_bid_low": row["top_page_bid_low"],
                    "geo_priority_score": row["geo_priority_score"],
                    "volume_rank": row["volume_rank"],
                    "cumulative_volume_share": row["cumulative_volume_share"],
                    "is_top_20pct_by_row": row["is_top_20pct_by_row"],
                    "is_top_80pct_volume_cover": row["is_top_80pct_volume_cover"],
                }
            )

            counter += 1

    questions_df = pd.DataFrame(rows)

    if questions_df.empty:
        return questions_df

    questions_df = questions_df.drop_duplicates(
        subset=["region", "question"]
    ).reset_index(drop=True)

    questions_df["question_id"] = [
        f"KQ{i:05d}" for i in range(1, len(questions_df) + 1)
    ]

    return questions_df


def print_summary(
    all_keywords: pd.DataFrame,
    selected_keywords: pd.DataFrame,
    questions_df: pd.DataFrame,
) -> None:
    print("=" * 90)
    print("KEYWORD-BASED GEO QUESTION GENERATOR")
    print("=" * 90)

    print("\nAll keyword rows:")
    print(all_keywords.groupby("region").size())

    print("\nAll keyword search volume:")
    print(all_keywords.groupby("region")["avg_monthly_searches"].sum().astype(int))

    print("\nRelevant keyword rows:")
    print(all_keywords[all_keywords["is_relevant"]].groupby("region").size())

    print("\nTop 20% rows by region:")
    print(all_keywords[all_keywords["is_top_20pct_by_row"]].groupby("region").size())

    print("\nSelected keywords for question generation:")
    print(selected_keywords.groupby("region").size())

    print("\nSelected keyword search volume:")
    print(selected_keywords.groupby("region")["avg_monthly_searches"].sum().astype(int))

    print("\nGenerated questions:")
    if questions_df.empty:
        print("No questions generated.")
    else:
        print(questions_df.groupby("region").size())

    print("\nTop selected topics:")
    topic_summary = (
        selected_keywords.groupby(["region", "topic"])
        .agg(
            keyword_count=("keyword", "count"),
            total_searches=("avg_monthly_searches", "sum"),
            avg_priority=("geo_priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["region", "total_searches"], ascending=[True, False])
    )

    if topic_summary.empty:
        print("No topic summary.")
    else:
        print(topic_summary.head(50).to_string(index=False))

    print("\nEstimated question volume if expanded:")
    top20_count = len(
        all_keywords[
            all_keywords["is_top_20pct_by_row"]
            & all_keywords["is_relevant"]
        ]
    )

    print(f"Relevant top-20%-row keywords: {top20_count}")
    print(f"If 6 templates each: {top20_count * 6:,} questions")
    print(f"If 8 templates each: {top20_count * 8:,} questions")
    print(f"If repeated 3 times: {top20_count * 6 * 3:,} API calls for 6-template design")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GEO questions from keyword CSV files."
    )

    parser.add_argument("--max-nz", type=int, default=300)
    parser.add_argument("--max-au", type=int, default=100)
    parser.add_argument("--max-ca", type=int, default=100)
    parser.add_argument("--templates", type=int, default=6)
    parser.add_argument("--write-questions", action="store_true")

    args = parser.parse_args()

    if args.templates < 1:
        raise ValueError("--templates must be at least 1")

    if args.templates > len(QUESTION_TEMPLATES):
        raise ValueError(
            f"--templates cannot exceed {len(QUESTION_TEMPLATES)}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    keyword_frames = []

    for region, path in DEFAULT_FILES.items():
        print(f"Loading {region}: {path}")
        keyword_frames.append(load_keyword_file(region, path))

    all_keywords = pd.concat(keyword_frames, ignore_index=True)
    all_keywords = add_priority_scores(all_keywords)

    selected_keywords = select_keywords(
        all_keywords,
        max_nz=args.max_nz,
        max_au=args.max_au,
        max_ca=args.max_ca,
        only_relevant=True,
    )

    questions_df = generate_questions(
        selected_keywords=selected_keywords,
        max_templates_per_keyword=args.templates,
    )

    all_keywords_path = OUTPUTS_DIR / "keyword_master_scored.csv"
    selected_keywords_path = OUTPUTS_DIR / "keyword_selected_for_geo.csv"
    questions_full_path = DATA_DIR / "geo_questions_from_keywords.csv"
    questions_run_path = DATA_DIR / "questions_from_keywords_run_ready.csv"
    topic_summary_path = OUTPUTS_DIR / "keyword_topic_summary.csv"

    all_keywords.to_csv(all_keywords_path, index=False)
    selected_keywords.to_csv(selected_keywords_path, index=False)
    questions_df.to_csv(questions_full_path, index=False)

    run_ready_cols = ["question_id", "region", "topic", "category", "question"]

    if not questions_df.empty:
        questions_df[run_ready_cols].to_csv(questions_run_path, index=False)
    else:
        pd.DataFrame(columns=run_ready_cols).to_csv(questions_run_path, index=False)

    topic_summary = (
        selected_keywords.groupby(["region", "topic", "category"])
        .agg(
            keyword_count=("keyword", "count"),
            total_searches=("avg_monthly_searches", "sum"),
            avg_priority=("geo_priority_score", "mean"),
        )
        .reset_index()
        .sort_values(["region", "total_searches"], ascending=[True, False])
    )

    topic_summary.to_csv(topic_summary_path, index=False)

    print_summary(all_keywords, selected_keywords, questions_df)

    print("\nSaved files:")
    print(f"- {all_keywords_path}")
    print(f"- {selected_keywords_path}")
    print(f"- {questions_full_path}")
    print(f"- {questions_run_path}")
    print(f"- {topic_summary_path}")

    if args.write_questions:
        final_questions_path = DATA_DIR / "questions.csv"

        if not questions_df.empty:
            questions_df[run_ready_cols].to_csv(final_questions_path, index=False)
        else:
            pd.DataFrame(columns=run_ready_cols).to_csv(final_questions_path, index=False)

        print(f"\nWrote run-ready questions to: {final_questions_path}")
    else:
        print("\nPreview mode only. data/questions.csv was NOT overwritten.")
        print("To overwrite data/questions.csv, rerun with --write-questions")


if __name__ == "__main__":
    main()