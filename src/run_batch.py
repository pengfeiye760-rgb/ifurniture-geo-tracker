import argparse
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
    max_questions: int | None = 5,
    repetitions: int = 1,
    use_web_search: bool = True,
    sleep_seconds: float = 1.0,
) -> None:
    init_db()

    questions = load_questions()

    if max_questions is not None:
        questions = questions[:max_questions]

    run_name = (
        f"geo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_q{len(questions)}_r{repetitions}"
    )

    run_id = create_run(
        run_name=run_name,
        model=DEFAULT_MODEL,
        region="MULTI",
        notes=(
            f"GEO batch run. questions={len(questions)}; "
            f"repetitions={repetitions}; use_web_search={use_web_search}"
        ),
    )

    total_calls = len(questions) * repetitions

    print(f"Created run_id: {run_id}")
    print(f"Run name: {run_name}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Questions to run: {len(questions)}")
    print(f"Repetitions per question: {repetitions}")
    print(f"Total API calls: {total_calls}")
    print(f"Web search enabled: {use_web_search}")
    print("=" * 80)

    call_index = 0

    for q_idx, q in enumerate(questions, start=1):
        for rep in range(1, repetitions + 1):
            call_index += 1

            stored_question_row = q.copy()

            if repetitions > 1:
                stored_question_row["question_id"] = f"{q['question_id']}_S{rep}"
            else:
                stored_question_row["question_id"] = q["question_id"]

            print(
                f"\n[{call_index}/{total_calls}] "
                f"{q['question_id']} sample {rep}/{repetitions} - {q['question']}"
            )

            try:
                bundle = collect_answer_bundle(
                    question=q["question"],
                    region=q["region"],
                    model=DEFAULT_MODEL,
                    use_web_search=use_web_search,
                )

                raw_answer = bundle["raw_answer"]
                features = extract_answer_features(raw_answer)

                if use_web_search and bundle.get("api_sources"):
                    features["sources"] = bundle["api_sources"]
                else:
                    features.setdefault("sources", [])

                answer_id = insert_answer(
                    run_id=run_id,
                    question_row=stored_question_row,
                    model=DEFAULT_MODEL,
                    raw_answer=raw_answer,
                    features=features,
                )

                insert_sources(
                    answer_id=answer_id,
                    question_id=stored_question_row["question_id"],
                    region=q["region"],
                    sources=features.get("sources", []),
                )

                print("Saved answer_id:", answer_id)
                print("Region:", q["region"])
                print("Topic:", q["topic"])
                print("Sample:", rep)
                print("iFurniture mentioned:", features["ifurniture_mentioned"])
                print("iFurniture rank:", features["ifurniture_rank"])
                print("Sentiment:", features["ifurniture_sentiment"])
                print("Risk mentioned:", features["risk_mentioned"])
                print("Brands:", features["brands"])
                print("Sources found:", len(features["sources"]))

                if features["sources"]:
                    print("Source domains:")
                    for source in features["sources"][:8]:
                        print(" -", source.get("domain", ""), "|", source.get("url", ""))

                    if len(features["sources"]) > 8:
                        print(f" ... and {len(features['sources']) - 8} more sources")

            except Exception as e:
                print("ERROR:", repr(e))

            time.sleep(sleep_seconds)

    print("\nBatch run finished.")
    print(f"Completed run_id: {run_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEO tracker batch collection.")

    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Number of questions to run. Use 100 for the full question bank.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of samples per question.",
    )

    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch(
        max_questions=args.max_questions,
        repetitions=args.repetitions,
        use_web_search=not args.no_web_search,
        sleep_seconds=args.sleep_seconds,
    )