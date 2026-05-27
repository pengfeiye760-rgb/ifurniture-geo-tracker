from pathlib import Path
import os
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

QUESTIONS_CSV = DATA_DIR / "questions.csv"
DB_PATH = DATA_DIR / "geo_tracker.db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


def has_openai_key() -> bool:
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY != "your_api_key_here"


def require_openai_key() -> str:
    if not has_openai_key():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Please add it to your .env file."
        )
    return OPENAI_API_KEY


def mask_key(key: str) -> str:
    if len(key) < 12:
        return "***"
    return key[:7] + "..." + key[-4:]


if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("QUESTIONS_CSV exists:", QUESTIONS_CSV.exists())
    print("DB_PATH:", DB_PATH)
    print("ENV_PATH exists:", ENV_PATH.exists())

    if has_openai_key():
        print("OPENAI_API_KEY loaded:", True)
        print("OPENAI_API_KEY preview:", mask_key(OPENAI_API_KEY))
    else:
        print("OPENAI_API_KEY loaded:", False)
        print("Please update your .env file.")