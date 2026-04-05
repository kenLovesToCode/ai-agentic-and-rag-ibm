"""
Basic LangChain + Gemini API example that prefers a free-tier Gemma model.

Setup:
    pip install -U langchain-core langchain-google-genai python-dotenv

Environment:
    Put your key in a local .env file:
        GOOGLE_API_KEY="your_google_ai_studio_key"

Optional:
    GOOGLE_MODEL="gemma-4"  # force a specific model if you already know it

Run:
    python3 gen1gemma.py
    python3 gen1gemma.py "Explain RAG in one short paragraph."
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


MODELS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GOOGLE_API_KEY.\n"
            "Create one in Google AI Studio, then run:\n"
            'export GOOGLE_API_KEY="your_key_here"'
        )
    return api_key


def fetch_models(api_key: str) -> list[dict]:
    query = urllib.parse.urlencode({"key": api_key, "pageSize": 1000})
    request = urllib.request.Request(f"{MODELS_ENDPOINT}?{query}")

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Could not list Gemini API models: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while listing Gemini API models: {exc}") from exc

    return payload.get("models", [])


def normalize_model_name(model: dict) -> str:
    if model.get("baseModelId"):
        return model["baseModelId"]
    return model.get("name", "").removeprefix("models/")


def supports_generate_content(model: dict) -> bool:
    methods = model.get("supportedGenerationMethods", [])
    return "generateContent" in methods


def choose_model(models: list[dict]) -> str:
    override = os.getenv("GOOGLE_MODEL")
    if override:
        return override

    candidates = []
    for model in models:
        if not supports_generate_content(model):
            continue
        base_model = normalize_model_name(model)
        description = (model.get("description") or "").lower()
        display_name = (model.get("displayName") or "").lower()
        combined = f"{base_model} {display_name} {description}".lower()
        if "deprecated" in description:
            continue
        candidates.append((base_model, combined))

    preference_order = [
        lambda name, text: name.startswith("gemma-4") or "gemma 4" in text,
        lambda name, text: name.startswith("gemma-3") or "gemma 3" in text,
        lambda name, text: name == "gemini-2.5-flash-lite-preview-09-2025",
        lambda name, text: name == "gemini-2.5-flash-lite",
        lambda name, text: name == "gemini-2.5-flash",
        lambda name, text: name == "gemini-2.0-flash-lite",
        lambda name, text: name == "gemini-2.0-flash",
    ]

    for matcher in preference_order:
        for name, text in candidates:
            if matcher(name, text):
                return name

    if candidates:
        return candidates[0][0]

    raise RuntimeError("No Gemini API chat model with generateContent support was found.")


def build_chain(api_key: str, model_name: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful beginner-friendly AI assistant. "
                "Answer clearly and keep it concise.",
            ),
            ("human", "{user_input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )

    return prompt | llm | StrOutputParser()


def main() -> None:
    load_dotenv()
    api_key = get_api_key()
    models = fetch_models(api_key)
    model_name = choose_model(models)
    user_input = (
        " ".join(sys.argv[1:]).strip()
        or "Explain what Test Driven Development (TDD) is in 3 simple bullet points."
    )

    chain = build_chain(api_key, model_name)
    response = chain.invoke({"user_input": user_input})

    print(f"Using model: {model_name}\n")
    print(response)


if __name__ == "__main__":
    main()
