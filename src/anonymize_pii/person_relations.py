"""Relation-aware PERSON tag replacement using context inference via local Ollama."""

import json
import re
from collections import Counter
from typing import Any

import requests


PERSON_TAG_PATTERN = re.compile(r"<PERSON_(\d+)>")
PROMPT_TARGETS_PLACEHOLDER = "[[TARGETS_JSON]]"


def _require_key(mapping: dict[str, Any], key: str) -> Any:
    """Return mapping[key] or raise a clear ValueError for missing config."""
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _parse_person_index(tag: str | None) -> int | None:
    """Return PERSON index from a tag like <PERSON_12>, else None."""
    match = PERSON_TAG_PATTERN.fullmatch(tag or "")
    return int(match.group(1)) if match else None


def _ordered_person_tags_in_text(anonymized_text: str) -> list[str]:
    """Collect unique PERSON tags in first-appearance order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for match in PERSON_TAG_PATTERN.finditer(anonymized_text or ""):
        tag = match.group(0)
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _source_text_by_tag(entity_mapping: dict[str, Any] | None) -> dict[str, str]:
    """Invert EntityMapping[PERSON]: source_text -> tag into tag -> source_text."""
    lookup: dict[str, str] = {}
    person_map = (entity_mapping or {}).get("PERSON", {})
    for source_text, tag in person_map.items():
        if _parse_person_index(tag) is not None:
            lookup[tag] = source_text
    return lookup


def _find_context_span(
    text: str,
    source_text: str | None,
    window_chars: int,
) -> tuple[int | None, int | None, str]:
    """Find first source_text occurrence and return bounded context around it."""
    if not text or not source_text:
        return None, None, ""

    start = text.find(source_text)
    if start < 0:
        return None, None, ""

    end = start + len(source_text)
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return start, end, text[left:right]


def _build_batch_relation_prompt(
    target_rows: list[dict[str, Any]],
    relation_config: dict[str, Any],
) -> str:
    """Build one batch prompt using the configured template and target payload."""
    target_payload: list[dict[str, str]] = []
    for row in target_rows:
        target_payload.append(
            {
                "person_tag": str(row["original_tag"]),
                "source_text": str(row.get("source_text")),
                "context": str(row.get("context")),
            }
        )

    targets_json = json.dumps(target_payload, ensure_ascii=True, separators=(",", ":"))
    template = _require_key(relation_config, "batch_prompt_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Config key 'batch_prompt_template' must be a non-empty string")
    if PROMPT_TARGETS_PLACEHOLDER not in template:
        raise ValueError(
            "Config key 'batch_prompt_template' must include [[TARGETS_JSON]] placeholder"
        )
    return template.replace(PROMPT_TARGETS_PLACEHOLDER, targets_json)


def _call_ollama(prompt: str, relation_config: dict[str, Any]) -> dict[str, Any]:
    """Call Ollama /api/generate and parse the top-level JSON object from response."""
    ollama_cfg = _require_key(relation_config, "ollama")
    if not isinstance(ollama_cfg, dict):
        raise ValueError("Config key 'ollama' must be a mapping")

    model = _require_key(ollama_cfg, "model")
    url = _require_key(ollama_cfg, "url")
    timeout_seconds = _require_key(ollama_cfg, "timeout_seconds")
    temperature = _require_key(ollama_cfg, "temperature")

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        url,
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    body = response.json()

    if "response" not in body:
        raise ValueError("Missing response field from Ollama")

    text = (body.get("response") or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError("Model output does not contain JSON object")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object")
    return parsed


def _parse_batch_assignments(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the assignments list from parsed model output."""
    assignments = parsed.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Batch response missing assignments list")
    return [item for item in assignments if isinstance(item, dict)]


def _normalize_relation_label(raw_label: str | None) -> str | None:
    """Normalize free-text relation labels into PATIENT_* replacement bases."""
    label = (raw_label or "").strip().lower()
    if not label or label == "unknown":
        return None

    normalized = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    if not normalized:
        return None

    if normalized in {"patient", "self"}:
        return "PATIENT"

    if not normalized.startswith("patient_"):
        normalized = f"patient_{normalized}"

    return normalized.upper()


def _parse_bool(value: Any) -> bool | None:
    """Parse flexible truthy/falsy values from model output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _apply_numbered_generated_tags(rows: list[dict[str, Any]]) -> None:
    """Apply _1/_2 numbering to model-generated replacement tag bases."""
    seen: Counter[str] = Counter()
    for row in rows:
        if not row.get("generated_by_model"):
            continue
        replacement_tag = row.get("replacement_tag")
        if not replacement_tag:
            continue
        base = replacement_tag.strip("<>")
        seen[base] += 1
        row["replacement_tag"] = f"<{base}_{seen[base]}>"


def _apply_replacements_in_text(anonymized_text: str, rows: list[dict[str, Any]]) -> str:
    """Replace original PERSON tags with computed replacements in anonymized text."""
    updated_text = anonymized_text
    for row in rows:
        original_tag = row["original_tag"]
        replacement_tag = row.get("replacement_tag")
        if not replacement_tag:
            continue
        if replacement_tag != original_tag:
            updated_text = updated_text.replace(original_tag, replacement_tag)
    return updated_text


def extract_and_apply_person_relations(
    original_text: str,
    anonymized_text: str,
    entity_mapping: dict[str, Any],
    relation_config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve PERSON tags to relation-aware tags and apply replacements.

    Returns a tuple of:
    1) updated anonymized text
    2) per-tag mapping rows used for auditing and troubleshooting
    """
    ordered_tags = _ordered_person_tags_in_text(anonymized_text)
    if not ordered_tags:
        return anonymized_text, []

    source_lookup = _source_text_by_tag(entity_mapping)

    max_persons = int(_require_key(relation_config, "max_persons_per_report"))
    confidence_threshold = float(_require_key(relation_config, "confidence_threshold"))
    window_chars = int(_require_key(relation_config, "context_window_chars"))

    selected_tags = ordered_tags[:max_persons]
    rows: list[dict[str, Any]] = []
    llm_target_rows: list[dict[str, Any]] = []

    for person_tag in selected_tags:
        person_idx = _parse_person_index(person_tag)
        person_text = source_lookup.get(person_tag)
        start, end, context = _find_context_span(original_text, person_text, window_chars)

        row: dict[str, Any] = {
            "person_index": person_idx,
            "source_text": person_text,
            "original_tag": person_tag,
            "replacement_tag": person_tag,
            "relation_label_raw": None,
            "rationale_short": None,
            "confidence": 0.0,
            "status": "fallback_no_context",
            "context_start": start,
            "context_end": end,
            "generated_by_model": False,
        }

        if not context:
            rows.append(row)
            continue

        row["context"] = context
        llm_target_rows.append(row)
        rows.append(row)

    if llm_target_rows:
        batch_size = max(1, int(_require_key(relation_config, "batch_size")))
        for i in range(0, len(llm_target_rows), batch_size):
            chunk = llm_target_rows[i : i + batch_size]
            prompt = _build_batch_relation_prompt(chunk, relation_config)

            try:
                parsed = _call_ollama(prompt, relation_config)
                assignments = _parse_batch_assignments(parsed)
                by_tag = {
                    item.get("person_tag"): item
                    for item in assignments
                    if item.get("person_tag")
                }

                for row in chunk:
                    item = by_tag.get(row["original_tag"])
                    if item is None:
                        row["status"] = "fallback_parse_error"
                        continue

                    raw_label = item.get("relation_label")
                    related = _parse_bool(item.get("related_to_patient"))
                    confidence = float(item.get("confidence", 0.0))
                    normalized = _normalize_relation_label(raw_label)
                    rationale_short = item.get("rationale_short")

                    row["relation_label_raw"] = raw_label
                    row["rationale_short"] = rationale_short
                    row["confidence"] = confidence

                    if related is False:
                        row["status"] = "unrelated_kept"
                    elif normalized and confidence >= confidence_threshold:
                        row["replacement_tag"] = f"<{normalized}>"
                        row["generated_by_model"] = True
                        row["status"] = "resolved"
                    else:
                        row["status"] = "fallback_low_confidence"
            except Exception as exc:
                for row in chunk:
                    row["status"] = "fallback_request_error"
                    row["error"] = str(exc)

    for row in rows:
        row.pop("context", None)

    _apply_numbered_generated_tags(rows)
    updated_text = _apply_replacements_in_text(anonymized_text, rows)
    return updated_text, rows
