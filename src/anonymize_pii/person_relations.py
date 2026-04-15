"""Relation-aware PERSON tag replacement using structured provider-agnostic LLM calls."""

import json
import os
import re
from collections import Counter
from typing import Any

from any_llm import completion
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


PERSON_TAG_PATTERN = re.compile(r"<PERSON_(\d+)>")
PROMPT_TARGETS_PLACEHOLDER = "[[TARGETS_JSON]]"
MAX_RELATION_RETRIES = 1


class RelationRequestError(RuntimeError):
    """Raised when an LLM request cannot be completed."""


class RelationParseError(ValueError):
    """Raised when structured LLM output cannot be validated."""


class RelationProviderConfig(BaseModel):
    """LLM provider settings for person relation extraction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class RelationRuntimeConfig(BaseModel):
    """Top-level relation runtime settings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    enabled_for_mask: str = Field(min_length=1)
    llm: RelationProviderConfig
    batch_system_prompt: str = Field(min_length=1)
    batch_user_prompt_template: str = Field(min_length=1)
    context_window_chars: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    max_persons_per_report: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class RelationTargetPayload(BaseModel):
    """Per-target payload sent to the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    person_tag: str
    source_text: str
    context: str

    @field_validator("person_tag")
    @classmethod
    def _validate_person_tag(cls, value: str) -> str:
        if _parse_person_index(value) is None:
            raise ValueError("person_tag must match <PERSON_N>")
        return value


class RelationAssignment(BaseModel):
    """One relation assignment returned by the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    person_tag: str
    relation_label: str = Field(min_length=1)
    related_to_patient: bool = Field(strict=True)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        min_length=1,
        description=(
            "Informative rationale explaining the relation decision using evidence from context"
        ),
    )

    @field_validator("person_tag")
    @classmethod
    def _validate_person_tag(cls, value: str) -> str:
        if _parse_person_index(value) is None:
            raise ValueError("person_tag must match <PERSON_N>")
        return value


class RelationBatchResponse(BaseModel):
    """Structured relation response from the LLM."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[RelationAssignment]


def _load_relation_runtime_config(relation_config: dict[str, Any]) -> RelationRuntimeConfig:
    """Validate relation config once and enforce required prompt placeholders."""
    config = RelationRuntimeConfig.model_validate(relation_config)
    if PROMPT_TARGETS_PLACEHOLDER not in config.batch_user_prompt_template:
        raise ValueError(
            "Config key 'batch_user_prompt_template' must include [[TARGETS_JSON]] placeholder"
        )
    return config


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


def _build_batch_relation_messages(
    target_rows: list[dict[str, Any]],
    relation_config: RelationRuntimeConfig,
    retry_error: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages with separate system and user prompts."""
    target_payload: list[RelationTargetPayload] = []
    for row in target_rows:
        target_payload.append(
            RelationTargetPayload(
                person_tag=str(row["original_tag"]),
                source_text=str(row.get("source_text")),
                context=str(row.get("context")),
            )
        )

    targets_json = json.dumps(
        [item.model_dump(mode="json") for item in target_payload],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    user_prompt = relation_config.batch_user_prompt_template.replace(
        PROMPT_TARGETS_PLACEHOLDER, targets_json
    )

    if retry_error:
        expected_tags = [str(row["original_tag"]) for row in target_rows]
        expected_tags_json = json.dumps(expected_tags, ensure_ascii=True, separators=(",", ":"))
        user_prompt = (
            f"{user_prompt}\n\n"
            "Your prior response was invalid JSON/schema for this task. "
            f"Validation error: {retry_error}\n"
            f"Return exactly one assignment for each of these person_tag values: {expected_tags_json}."
        )

    return [
        {"role": "system", "content": relation_config.batch_system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object_text(text: str) -> str:
    """Extract first top-level JSON object boundaries from raw text."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError("Model output does not contain JSON object")
    return text[start : end + 1]


def _extract_message_text(content: Any) -> str:
    """Normalize provider message content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments)

    return ""


def _extract_parsed_relation_response(response: Any) -> RelationBatchResponse:
    """Parse completion output into validated RelationBatchResponse."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise RelationParseError("LLM response missing choices")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RelationParseError("LLM response missing message")

    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, RelationBatchResponse):
        return parsed

    if isinstance(parsed, dict):
        try:
            return RelationBatchResponse.model_validate(parsed)
        except ValidationError as exc:
            raise RelationParseError(f"Invalid parsed response: {exc}") from exc

    text = _extract_message_text(getattr(message, "content", ""))
    json_text = _extract_json_object_text(text.strip())
    try:
        return RelationBatchResponse.model_validate_json(json_text)
    except ValidationError as exc:
        raise RelationParseError(f"Invalid JSON response: {exc}") from exc


def _call_relation_model(
    messages: list[dict[str, str]],
    relation_config: RelationRuntimeConfig,
) -> RelationBatchResponse:
    """Call configured provider through any-llm and return validated response model."""
    llm_cfg = relation_config.llm

    api_key = llm_cfg.api_key
    if llm_cfg.api_key_env:
        api_key = os.getenv(llm_cfg.api_key_env, api_key)

    call_kwargs: dict[str, Any] = {
        "model": llm_cfg.model,
        "provider": llm_cfg.provider,
        "messages": messages,
        "temperature": llm_cfg.temperature,
    }

    if llm_cfg.max_tokens is not None:
        call_kwargs["max_tokens"] = llm_cfg.max_tokens
    if llm_cfg.api_base:
        call_kwargs["api_base"] = llm_cfg.api_base
    if api_key:
        call_kwargs["api_key"] = api_key

    call_kwargs["client_args"] = {"timeout": llm_cfg.timeout_seconds}

    try:
        response = completion(response_format=RelationBatchResponse, **call_kwargs)
    except Exception:
        try:
            response = completion(**call_kwargs)
        except Exception as exc:
            raise RelationRequestError(f"LLM request failed: {exc}") from exc

    try:
        return _extract_parsed_relation_response(response)
    except RelationParseError:
        raise
    except Exception as exc:
        raise RelationParseError(f"Could not parse response: {exc}") from exc


def _assignments_by_tag(
    batch_response: RelationBatchResponse,
    expected_tags: list[str],
) -> dict[str, RelationAssignment]:
    """Map validated assignments by person_tag and enforce exact coverage."""
    by_tag: dict[str, RelationAssignment] = {}

    for assignment in batch_response.assignments:
        if assignment.person_tag in by_tag:
            raise RelationParseError(f"Duplicate assignment for {assignment.person_tag}")
        by_tag[assignment.person_tag] = assignment

    expected = set(expected_tags)
    missing = [tag for tag in expected_tags if tag not in by_tag]
    extras = sorted(tag for tag in by_tag if tag not in expected)
    if missing or extras:
        raise RelationParseError(
            f"Assignment coverage mismatch. missing={missing}, extras={extras}"
        )

    return by_tag


def _infer_chunk_assignments(
    chunk: list[dict[str, Any]],
    relation_config: RelationRuntimeConfig,
) -> dict[str, RelationAssignment]:
    """Run relation inference with one retry on request/parse failures."""
    last_exc: Exception | None = None

    for attempt in range(MAX_RELATION_RETRIES + 1):
        retry_error = str(last_exc) if attempt > 0 and last_exc else None
        messages = _build_batch_relation_messages(chunk, relation_config, retry_error=retry_error)
        expected_tags = [str(row["original_tag"]) for row in chunk]

        try:
            parsed_response = _call_relation_model(messages, relation_config)
            return _assignments_by_tag(parsed_response, expected_tags)
        except (RelationParseError, RelationRequestError) as exc:
            last_exc = exc

    if last_exc is None:
        raise RelationRequestError("Relation inference failed without an exception")
    raise last_exc


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

    runtime_config = _load_relation_runtime_config(relation_config)

    source_lookup = _source_text_by_tag(entity_mapping)

    max_persons = runtime_config.max_persons_per_report
    confidence_threshold = runtime_config.confidence_threshold
    window_chars = runtime_config.context_window_chars

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
            "rationale": None,
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
        batch_size = max(1, int(runtime_config.batch_size))
        for i in range(0, len(llm_target_rows), batch_size):
            chunk = llm_target_rows[i : i + batch_size]

            try:
                by_tag = _infer_chunk_assignments(chunk, runtime_config)

                for row in chunk:
                    item = by_tag[row["original_tag"]]

                    raw_label = item.relation_label
                    related = item.related_to_patient
                    confidence = item.confidence
                    normalized = _normalize_relation_label(raw_label)
                    rationale = item.rationale

                    row["relation_label_raw"] = raw_label
                    row["rationale"] = rationale
                    row["confidence"] = confidence

                    if not related:
                        row["replacement_tag"] = row["original_tag"]
                        row["status"] = "unrelated_kept"
                    elif normalized and confidence >= confidence_threshold:
                        row["replacement_tag"] = normalized
                        row["generated_by_model"] = True
                        row["status"] = "resolved"
                    else:
                        row["status"] = "fallback_low_confidence"
            except RelationParseError as exc:
                for row in chunk:
                    row["status"] = "fallback_parse_error"
                    row["error"] = str(exc)
            except RelationRequestError as exc:
                for row in chunk:
                    row["status"] = "fallback_request_error"
                    row["error"] = str(exc)

    for row in rows:
        row.pop("context", None)

    _apply_numbered_generated_tags(rows)
    updated_text = _apply_replacements_in_text(anonymized_text, rows)
    return updated_text, rows
