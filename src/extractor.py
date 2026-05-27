import json
import re
from typing import Any, Dict, List, Optional

import tldextract


KNOWN_BRANDS = [
    "iFurniture",
    "Target Furniture",
    "Freedom Furniture",
    "Oak Furniture Store",
    "The Warehouse",
    "Mocka",
    "Idiya",
    "Tsumbay",
    "Bedpost",
    "Bedroom Store",
    "BedsRus",
    "Hunter Home",
    "Nood",
    "Danske Møbler",
    "Trade Me",
    "Kmart",
    "IKEA",
]

IFURNITURE_PATTERNS = [
    r"\bifurniture\b",
    r"\bi-furniture\b",
    r"\bi furniture\b",
    r"\bifurniture\.co\.nz\b",
]

RISK_PHRASES = [
    "mixed reviews",
    "mixed feedback",
    "delivery delay",
    "delivery delays",
    "delays",
    "damaged items",
    "damaged item",
    "damage",
    "poor delivery",
    "slow delivery",
    "late delivery",
    "customer service issues",
    "manage expectations",
    "quality issues",
    "complaints",
    "negative reviews",
]

POSITIVE_WORDS = [
    "affordable",
    "good value",
    "best value",
    "budget-friendly",
    "recommended",
    "reliable",
    "large showroom",
    "wide range",
    "lowest price",
    "great value",
    "popular",
    "convenient",
    "in-house delivery",
]

NEGATIVE_WORDS = [
    "mixed reviews",
    "delays",
    "damaged",
    "complaints",
    "poor",
    "negative",
    "unreliable",
    "quality issues",
    "manage expectations",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def contains_ifurniture(text: str) -> bool:
    lower_text = text.lower()
    return any(re.search(pattern, lower_text, flags=re.IGNORECASE) for pattern in IFURNITURE_PATTERNS)


def extract_brands(text: str) -> List[str]:
    found = []
    lower_text = text.lower()

    for brand in KNOWN_BRANDS:
        if brand.lower() in lower_text:
            found.append(brand)

    return found


def estimate_ifurniture_rank(text: str) -> Optional[int]:
    """
    Estimate iFurniture ranking from numbered/bulleted recommendation lists.

    Examples:
    1. iFurniture
    2. Target Furniture

    or:
    - iFurniture
    - Freedom Furniture
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    ranked_items = []

    for line in lines:
        numbered_match = re.match(r"^(\d+)[\.\)]\s*(.+)", line)
        if numbered_match:
            rank = int(numbered_match.group(1))
            content = numbered_match.group(2)
            ranked_items.append((rank, content))
            continue

    for rank, content in ranked_items:
        if contains_ifurniture(content):
            return rank

    # Fallback: if no numbered list, estimate by first brand order in text.
    brand_positions = []
    lower_text = text.lower()

    for brand in KNOWN_BRANDS:
        pos = lower_text.find(brand.lower())
        if pos >= 0:
            brand_positions.append((pos, brand))

    brand_positions.sort()

    for idx, (_, brand) in enumerate(brand_positions, start=1):
        if brand.lower() == "ifurniture":
            return idx

    if contains_ifurniture(text):
        return None

    return None


def extract_risk_phrases(text: str) -> List[str]:
    lower_text = text.lower()
    found = []

    for phrase in RISK_PHRASES:
        if phrase.lower() in lower_text:
            found.append(phrase)

    return found


def estimate_ifurniture_sentiment(text: str) -> str:
    """
    Simple MVP rule:
    - If iFurniture is not mentioned: not_mentioned
    - Compare positive and negative phrase counts around whole answer.
    Later we can replace this with LLM structured extraction.
    """
    if not contains_ifurniture(text):
        return "not_mentioned"

    lower_text = text.lower()

    positive_count = sum(1 for word in POSITIVE_WORDS if word in lower_text)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in lower_text)

    if negative_count > positive_count:
        return "negative"
    if positive_count > negative_count:
        return "positive"
    return "neutral"


def extract_urls(text: str) -> List[str]:
    url_pattern = r"https?://[^\s\]\)\}>,]+"
    return re.findall(url_pattern, text)


def domain_from_url(url: str) -> str:
    extracted = tldextract.extract(url)
    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return extracted.domain


def extract_sources(text: str) -> List[Dict[str, Any]]:
    urls = extract_urls(text)
    sources = []

    for url in urls:
        sources.append(
            {
                "url": url,
                "domain": domain_from_url(url),
                "source_type": "unknown",
                "used_for": "",
                "sentiment_toward_ifurniture": "unknown",
            }
        )

    return sources


def extract_answer_features(raw_answer: str) -> Dict[str, Any]:
    text = normalize_text(raw_answer)

    brands = extract_brands(text)
    risk_phrases = extract_risk_phrases(text)
    sources = extract_sources(text)

    result = {
        "ifurniture_mentioned": contains_ifurniture(text),
        "ifurniture_rank": estimate_ifurniture_rank(raw_answer),
        "ifurniture_sentiment": estimate_ifurniture_sentiment(text),
        "risk_mentioned": bool(risk_phrases),
        "risk_phrases": risk_phrases,
        "brands": brands,
        "sources": sources,
    }

    return result


if __name__ == "__main__":
    sample_answer = """
    For affordable furniture in Auckland, good options include:

    1. iFurniture - often recommended for budget-friendly furniture, large showroom, and good value.
    2. Target Furniture - good for mid-range styles.
    3. Freedom Furniture - better for design-led options.

    However, some customers mention mixed reviews around delivery delays and damaged items.
    Source: https://www.ifurniture.co.nz/
    """

    features = extract_answer_features(sample_answer)
    print(json.dumps(features, indent=2, ensure_ascii=False))