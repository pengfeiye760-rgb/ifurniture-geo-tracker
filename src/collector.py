import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from openai import OpenAI

from config import require_openai_key


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


SYSTEM_PROMPT = """
You are a GEO monitoring assistant for a furniture retailer.

Your job is to answer consumer shopping questions as a normal AI assistant would,
but in a way that is useful for GEO tracking.

Rules:
- Answer naturally in English.
- Focus on the requested region.
- Mention relevant furniture retailers only if they are genuinely relevant.
- Do not force iFurniture into the answer if it is not relevant.
- If you mention brands, make the order meaningful.
- Keep the answer concise but informative.
- When web search is available, use sources and include factual, current information.
"""


def build_user_prompt(question: str, region: str = "NZ") -> str:
    region_hint = {
        "NZ": "New Zealand, especially Auckland if the question is location-specific",
        "AU": "Australia",
        "CA": "Canada",
    }.get(region.upper(), region)

    return f"""
Region: {region_hint}

Consumer question:
{question}

Please answer the question as a consumer-facing AI shopping assistant.
"""


def clean_url(url: str) -> str:
    """
    Remove tracking query parameters such as utm_source=openai.
    This helps deduplicate the same source URL.
    """
    try:
        parsed = urlparse(url)
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",
                "",
                "",
            )
        )
    except Exception:
        return url


def domain_from_url_simple(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain.replace("www.", "")
    except Exception:
        clean = url.replace("https://", "").replace("http://", "")
        return clean.split("/")[0].replace("www.", "")


def extract_api_sources(response: Any) -> List[Dict[str, str]]:
    """
    Extract only final answer citations from Responses API message annotations.

    Important:
    - We do NOT store all web_search_call.action.sources here.
    - Those are search-pool sources, not necessarily sources used in the final answer.
    - For GEO MVP, we only want cited sources that directly supported the AI answer.
    """
    sources = []

    try:
        data = response.model_dump()
    except Exception:
        return sources

    for output_item in data.get("output", []):
        if output_item.get("type") != "message":
            continue

        for content_item in output_item.get("content", []):
            annotations = content_item.get("annotations", []) or []

            for ann in annotations:
                url = ann.get("url")
                title = ann.get("title", "")

                if not url:
                    continue

                cleaned = clean_url(url)

                sources.append(
                    {
                        "url": cleaned,
                        "domain": domain_from_url_simple(cleaned),
                        "source_type": "api_citation",
                        "used_for": title,
                        "sentiment_toward_ifurniture": "unknown",
                    }
                )

    # Deduplicate by cleaned URL
    seen = set()
    unique_sources = []

    for source in sources:
        url = source.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_sources.append(source)

    return unique_sources


def collect_answer_bundle(
    question: str,
    region: str = "NZ",
    model: Optional[str] = None,
    use_web_search: bool = False,
) -> Dict[str, Any]:
    api_key = require_openai_key()
    client = OpenAI(api_key=api_key)

    selected_model = model or DEFAULT_MODEL

    request_kwargs = {
        "model": selected_model,
        "input": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(question, region),
            },
        ],
        "max_output_tokens": 1200,
    }

    if use_web_search:
        request_kwargs["tools"] = [{"type": "web_search"}]
        request_kwargs["tool_choice"] = "required"

    response = client.responses.create(**request_kwargs)

    return {
        "raw_answer": response.output_text,
        "api_sources": extract_api_sources(response),
    }


def collect_answer(
    question: str,
    region: str = "NZ",
    model: Optional[str] = None,
    use_web_search: bool = False,
) -> str:
    bundle = collect_answer_bundle(
        question=question,
        region=region,
        model=model,
        use_web_search=use_web_search,
    )
    return bundle["raw_answer"]


if __name__ == "__main__":
    test_question = "Where can I buy affordable furniture in Auckland?"

    print("Running one API test...")
    print("Question:", test_question)
    print("Model:", DEFAULT_MODEL)
    print("Web search enabled:", True)
    print("-" * 60)

    result = collect_answer_bundle(
        question=test_question,
        region="NZ",
        use_web_search=True,
    )

    print(result["raw_answer"])
    print("\nSources:")
    for source in result["api_sources"]:
        print("-", source)