import time
from datetime import datetime

import pandas as pd

from collector import collect_answer_bundle, DEFAULT_MODEL
from db import create_run, init_db, insert_answer, insert_sources
from extractor import extract_answer_features


def load_questions(path: str = "data/questions.csv") -> list[dict]:
    df = pd.read_csv(path)

    required_cols = ["question_id", "region", "topic", "category", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"questions.csv is missing columns: {missing_cols}")

    return df.to_dict(orient="records")


def run_batch(
    max_questions: int | None = 1,
    use_web_search: bool = True,
) -> None:
    init_db()

    questions = load_questions()

    if max_questions is not None:
        questions = questions[:max_questions]

    run_name = f"mvp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_id = create_run(
        run_name=run_name,
        model=DEFAULT_MODEL,
        region="NZ",
        notes=f"MVP batch run. use_web_search={use_web_search}",
    )

    print(f"Created run_id: {run_id}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Questions to run: {len(questions)}")
    print(f"Web search enabled: {use_web_search}")
    print("=" * 80)

    for idx, q in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] {q['question_id']} - {q['question']}")

        try:
            bundle = collect_answer_bundle(
                question=q["question"],
                region=q["region"],
                model=DEFAULT_MODEL,
                use_web_search=use_web_search,
            )

            raw_answer = bundle["raw_answer"]
            features = extract_answer_features(raw_answer)

            # Important:
            # If web_search=True, only use final API citations from collector.py.
            # Do NOT mix raw-answer URLs from extractor.py, because that can create duplicates:
            # e.g. /page?utm_source=openai and /page.
            if use_web_search and bundle.get("api_sources"):
                features["sources"] = bundle["api_sources"]
            else:
                features.setdefault("sources", [])

            answer_id = insert_answer(
                run_id=run_id,
                question_row=q,
                model=DEFAULT_MODEL,
                raw_answer=raw_answer,
                features=features,
            )

            insert_sources(
                answer_id=answer_id,
                question_id=q["question_id"],
                region=q["region"],
                sources=features.get("sources", []),
            )

            print("Saved answer_id:", answer_id)
            print("iFurniture mentioned:", features["ifurniture_mentioned"])
            print("iFurniture rank:", features["ifurniture_rank"])
            print("Sentiment:", features["ifurniture_sentiment"])
            print("Risk mentioned:", features["risk_mentioned"])
            print("Brands:", features["brands"])
            print("Sources found:", len(features["sources"]))

            if features["sources"]:
                print("Source domains:")
                for source in features["sources"]:
                    print(" -", source.get("domain", ""), "|", source.get("url", ""))

        except Exception as e:
            print("ERROR:", repr(e))

        time.sleep(1)

    print("\nBatch run finished.")


if __name__ == "__main__":
    run_batch(
        max_questions=10,
        use_web_search=True,
    )