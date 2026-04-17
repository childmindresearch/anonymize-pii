"""Relation-aware PERSON tag replacement using structured provider-agnostic LLM calls."""
import openai.types.shared.reasoning_effort

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
POSTPROCESS_ROWS_PLACEHOLDER = "[[POSTPROCESS_ROWS_JSON]]"
MAX_POSTPROCESS_RETRIES = 1
POSTPROCESS_TAG_BASE_PATTERN = re.compile(r"^PATIENT(?:_[A-Z0-9]+)*$")


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
    reasoning_effort: str = Field(default='none', min_length=1)
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
    postprocess_batch_size: int = Field(default=10, gt=0)
    postprocess_timeout_seconds: int | None = Field(default=90, gt=0)
    postprocess_system_prompt: str = Field(min_length=1)
    postprocess_user_prompt_template: str = Field(min_length=1)


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


class PostprocessTargetPayload(BaseModel):
    """Per-row payload sent to postprocessing LLM calls."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_tag: str
    current_tag: str = Field(min_length=1)
    source_text: str | None = None
    relation_label_raw: str | None = None
    rationale: str | None = None
    status: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    generated_by_model: bool = Field(strict=True)

    @field_validator("original_tag")
    @classmethod
    def _validate_original_tag(cls, value: str) -> str:
        if _parse_person_index(value) is None:
            raise ValueError("original_tag must match <PERSON_N>")
        return value


class PostprocessAssignment(BaseModel):
    """One standardized-tag decision returned by the postprocessing model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_tag: str
    keep_current_tag: bool = Field(strict=True)
    standardized_tag_base: str | None = None
    entity_group: str | None = None
    rationale: str = Field(min_length=1)

    @field_validator("original_tag")
    @classmethod
    def _validate_original_tag(cls, value: str) -> str:
        if _parse_person_index(value) is None:
            raise ValueError("original_tag must match <PERSON_N>")
        return value

    @field_validator("standardized_tag_base")
    @classmethod
    def _validate_standardized_base(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        if not cleaned:
            return None
        if not POSTPROCESS_TAG_BASE_PATTERN.fullmatch(cleaned):
            raise ValueError(
                "standardized_tag_base must match PATIENT or PATIENT_<RELATION> with no suffix"
            )
        return cleaned


class PostprocessBatchResponse(BaseModel):
    """Structured postprocessing response from the LLM."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[PostprocessAssignment]


def _load_relation_runtime_config(relation_config: dict[str, Any]) -> RelationRuntimeConfig:
    """Validate relation config once and enforce required prompt placeholders."""
    config = RelationRuntimeConfig.model_validate(relation_config)
    if PROMPT_TARGETS_PLACEHOLDER not in config.batch_user_prompt_template:
        raise ValueError(
            "Config key 'batch_user_prompt_template' must include [[TARGETS_JSON]] placeholder"
        )
    if POSTPROCESS_ROWS_PLACEHOLDER not in config.postprocess_user_prompt_template:
        raise ValueError(
            "Config key 'postprocess_user_prompt_template' must include [[POSTPROCESS_ROWS_JSON]] placeholder"
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


def _build_postprocess_messages(
    target_rows: list[dict[str, Any]],
    relation_config: RelationRuntimeConfig,
    document_rows: list[dict[str, Any]] | None = None,
    retry_error: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for document-level tag standardization."""
    payload_rows: list[PostprocessTargetPayload] = []
    for row in target_rows:
        payload_rows.append(
            PostprocessTargetPayload(
                original_tag=str(row["original_tag"]),
                current_tag=str(row.get("replacement_tag") or row["original_tag"]),
                source_text=row.get("source_text"),
                relation_label_raw=row.get("relation_label_raw"),
                rationale=row.get("rationale"),
                status=str(row.get("status") or "unknown"),
                confidence=float(row.get("confidence") or 0.0),
                generated_by_model=bool(row.get("generated_by_model")),
            )
        )

    rows_json = json.dumps(
        [item.model_dump(mode="json") for item in payload_rows],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    user_prompt = relation_config.postprocess_user_prompt_template.replace(
        POSTPROCESS_ROWS_PLACEHOLDER,
        rows_json,
    )

    if document_rows and len(document_rows) > len(target_rows):
        overview_payload = [
            {
                "original_tag": str(row["original_tag"]),
                "current_tag": str(row.get("replacement_tag") or row["original_tag"]),
                "source_text": str(row.get("source_text") or "")[:80],
                "relation_label_raw": str(row.get("relation_label_raw") or ""),
            }
            for row in document_rows
        ]
        overview_json = json.dumps(overview_payload, ensure_ascii=True, separators=(",", ":"))
        user_prompt = (
            f"{user_prompt}\n\n"
            "Document-wide rows overview for consistency across aliases and relations: "
            f"{overview_json}"
        )

    if retry_error:
        expected_tags = [str(row["original_tag"]) for row in target_rows]
        expected_tags_json = json.dumps(expected_tags, ensure_ascii=True, separators=(",", ":"))
        user_prompt = (
            f"{user_prompt}\n\n"
            "Your prior response was invalid JSON/schema for this task. "
            f"Validation error: {retry_error}\n"
            f"Return exactly one assignment for each of these original_tag values: {expected_tags_json}."
        )

    return [
        {"role": "system", "content": relation_config.postprocess_system_prompt},
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


def _extract_parsed_postprocess_response(response: Any) -> PostprocessBatchResponse:
    """Parse completion output into validated PostprocessBatchResponse."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise RelationParseError("LLM response missing choices")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RelationParseError("LLM response missing message")

    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, PostprocessBatchResponse):
        return parsed

    if isinstance(parsed, dict):
        try:
            return PostprocessBatchResponse.model_validate(parsed)
        except ValidationError as exc:
            raise RelationParseError(f"Invalid parsed response: {exc}") from exc

    text = _extract_message_text(getattr(message, "content", ""))
    json_text = _extract_json_object_text(text.strip())
    try:
        return PostprocessBatchResponse.model_validate_json(json_text)
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
        "reasoning_effort": llm_cfg.reasoning_effort,
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
            err = str(exc).strip() or exc.__class__.__name__
            raise RelationRequestError(f"LLM request failed: {err}") from exc

    try:
        return _extract_parsed_relation_response(response)
    except RelationParseError:
        raise
    except Exception as exc:
        raise RelationParseError(f"Could not parse response: {exc}") from exc


def _call_postprocess_model(
    messages: list[dict[str, str]],
    relation_config: RelationRuntimeConfig,
) -> PostprocessBatchResponse:
    """Call configured provider through any-llm for postprocessing decisions."""
    llm_cfg = relation_config.llm

    api_key = llm_cfg.api_key
    if llm_cfg.api_key_env:
        api_key = os.getenv(llm_cfg.api_key_env, api_key)

    call_kwargs: dict[str, Any] = {
        "model": llm_cfg.model,
        "provider": llm_cfg.provider,
        "messages": messages,
        "reasoning_effort": llm_cfg.reasoning_effort,
        "temperature": llm_cfg.temperature,
    }

    if llm_cfg.max_tokens is not None:
        call_kwargs["max_tokens"] = llm_cfg.max_tokens
    if llm_cfg.api_base:
        call_kwargs["api_base"] = llm_cfg.api_base
    if api_key:
        call_kwargs["api_key"] = api_key

    postprocess_timeout = relation_config.postprocess_timeout_seconds or llm_cfg.timeout_seconds
    call_kwargs["client_args"] = {"timeout": postprocess_timeout}

    try:
        response = completion(response_format=PostprocessBatchResponse, **call_kwargs)
    except Exception:
        try:
            response = completion(**call_kwargs)
        except Exception as exc:
            err = str(exc).strip() or exc.__class__.__name__
            raise RelationRequestError(f"LLM request failed: {err}") from exc

    try:
        return _extract_parsed_postprocess_response(response)
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


def _postprocess_assignments_by_tag(
    batch_response: PostprocessBatchResponse,
    expected_tags: list[str],
) -> dict[str, PostprocessAssignment]:
    """Map validated postprocessing assignments by original_tag with full coverage checks."""
    by_tag: dict[str, PostprocessAssignment] = {}

    for assignment in batch_response.assignments:
        if assignment.original_tag in by_tag:
            raise RelationParseError(f"Duplicate assignment for {assignment.original_tag}")
        by_tag[assignment.original_tag] = assignment

    expected = set(expected_tags)
    missing = [tag for tag in expected_tags if tag not in by_tag]
    extras = sorted(tag for tag in by_tag if tag not in expected)
    if missing or extras:
        raise RelationParseError(
            f"Assignment coverage mismatch. missing={missing}, extras={extras}"
        )

    for tag in expected_tags:
        assignment = by_tag[tag]
        if not assignment.keep_current_tag and not assignment.standardized_tag_base:
            raise RelationParseError(f"Missing standardized_tag_base for {tag}")
        if not assignment.keep_current_tag and not assignment.entity_group:
            raise RelationParseError(f"Missing entity_group for {tag}")

    return by_tag


def _infer_postprocess_assignments(
    target_rows: list[dict[str, Any]],
    relation_config: RelationRuntimeConfig,
) -> dict[str, PostprocessAssignment]:
    """Run document-level postprocessing inference in chunks with global context."""
    if not target_rows:
        return {}

    batch_size = max(1, int(relation_config.postprocess_batch_size))
    merged: dict[str, PostprocessAssignment] = {}

    for i in range(0, len(target_rows), batch_size):
        chunk = target_rows[i : i + batch_size]
        last_exc: Exception | None = None

        for attempt in range(MAX_POSTPROCESS_RETRIES + 1):
            retry_error = str(last_exc) if attempt > 0 and last_exc else None
            messages = _build_postprocess_messages(
                chunk,
                relation_config,
                document_rows=target_rows,
                retry_error=retry_error,
            )
            expected_tags = [str(row["original_tag"]) for row in chunk]

            try:
                parsed_response = _call_postprocess_model(messages, relation_config)
                by_tag = _postprocess_assignments_by_tag(parsed_response, expected_tags)
                merged.update(by_tag)
                break
            except (RelationParseError, RelationRequestError) as exc:
                last_exc = exc
        else:
            if last_exc is None:
                raise RelationRequestError("Postprocessing inference failed without an exception")
            raise last_exc

    return merged


def _normalize_postprocess_tag_base(raw_base: str | None) -> str | None:
    """Normalize postprocessing base tags into PATIENT/PATIENT_* form without counters."""
    candidate = (raw_base or "").strip()
    if not candidate:
        return None

    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1]

    candidate = re.sub(r"_\d+$", "", candidate)
    if POSTPROCESS_TAG_BASE_PATTERN.fullmatch(candidate):
        return candidate

    return _normalize_relation_label(candidate)


def _sanitize_entity_group(value: str | None, fallback: str) -> str:
    """Normalize model-suggested entity-group IDs for deterministic grouping."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned or fallback


def _relation_specificity_score(tag_base: str) -> tuple[int, int, int]:
    """Score tag-base specificity without hardcoding relation ontologies."""
    suffix = tag_base[len("PATIENT") :].lstrip("_") if tag_base.startswith("PATIENT") else ""
    if not suffix:
        return (0, 0, 0)
    parts = [part for part in suffix.split("_") if part]
    return (1, len(parts), len(suffix))


def _choose_group_tag_base(bases: list[str]) -> str:
    """Choose one canonical base per entity group using frequency then specificity."""
    counts = Counter(bases)
    max_count = max(counts.values())
    top = [base for base, count in counts.items() if count == max_count]
    return max(top, key=lambda item: (_relation_specificity_score(item), item))


def _build_standardized_tags_for_targets(
    target_rows: list[dict[str, Any]],
    assignments_by_tag: dict[str, PostprocessAssignment],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Build deterministic final tags from postprocessing assignments."""
    state_by_tag: dict[str, dict[str, Any]] = {}
    group_bases: dict[str, list[str]] = {}

    for idx, row in enumerate(target_rows):
        original_tag = str(row["original_tag"])
        current_tag = str(row.get("replacement_tag") or original_tag)
        assignment = assignments_by_tag[original_tag]

        normalized_base = _normalize_postprocess_tag_base(assignment.standardized_tag_base)
        fallback_group = f"tag_{_parse_person_index(original_tag) or idx + 1}"
        entity_group = _sanitize_entity_group(assignment.entity_group, fallback_group)

        if assignment.keep_current_tag or not normalized_base:
            state_by_tag[original_tag] = {
                "current_tag": current_tag,
                "keep_current_tag": True,
                "standardized_tag_base": None,
                "entity_group": None,
            }
            continue

        state_by_tag[original_tag] = {
            "current_tag": current_tag,
            "keep_current_tag": False,
            "standardized_tag_base": normalized_base,
            "entity_group": entity_group,
        }
        group_bases.setdefault(entity_group, []).append(normalized_base)

    group_to_base = {group: _choose_group_tag_base(bases) for group, bases in group_bases.items()}

    base_to_groups: dict[str, list[str]] = {}
    for row in target_rows:
        original_tag = str(row["original_tag"])
        state = state_by_tag[original_tag]
        if state["keep_current_tag"]:
            continue

        entity_group = str(state["entity_group"])
        base = group_to_base[entity_group]
        state["standardized_tag_base"] = base
        groups = base_to_groups.setdefault(base, [])
        if entity_group not in groups:
            groups.append(entity_group)

    final_tags_by_original: dict[str, str] = {}
    for row in target_rows:
        original_tag = str(row["original_tag"])
        state = state_by_tag[original_tag]
        current_tag = str(state["current_tag"])

        if state["keep_current_tag"]:
            final_tags_by_original[original_tag] = current_tag
            continue

        base = str(state["standardized_tag_base"])
        entity_group = str(state["entity_group"])
        groups = base_to_groups.get(base, [])
        if len(groups) <= 1:
            final_tags_by_original[original_tag] = f"<{base}>"
        else:
            final_tags_by_original[original_tag] = f"<{base}_{groups.index(entity_group) + 1}>"

    return final_tags_by_original, state_by_tag


def _apply_tag_mapping_in_text(anonymized_text: str, replacements: dict[str, str]) -> str:
    """Replace tags in one pass using exact-token mappings."""
    if not replacements:
        return anonymized_text

    pattern = re.compile(
        "|".join(re.escape(tag) for tag in sorted(replacements.keys(), key=len, reverse=True))
    )
    return pattern.sub(lambda match: replacements.get(match.group(0), match.group(0)), anonymized_text)


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


def standardize_person_relation_tags(
    anonymized_text: str,
    relation_rows: list[dict[str, Any]],
    relation_config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Postprocess relation-aware tags at document level without mutating relation_rows."""
    if not relation_rows:
        return anonymized_text, []

    runtime_config = _load_relation_runtime_config(relation_config)

    target_rows = [
        row
        for row in relation_rows
        if bool(row.get("generated_by_model"))
        and str(row.get("replacement_tag") or "").startswith("<PATIENT")
    ]
    target_tags = {str(row["original_tag"]) for row in target_rows}

    assignments_by_tag: dict[str, PostprocessAssignment] = {}
    state_by_tag: dict[str, dict[str, Any]] = {}
    final_tags_by_original: dict[str, str] = {}
    updated_text = anonymized_text
    fallback_status: str | None = None
    fallback_error: str | None = None

    if target_rows:
        try:
            assignments_by_tag = _infer_postprocess_assignments(target_rows, runtime_config)
            final_tags_by_original, state_by_tag = _build_standardized_tags_for_targets(
                target_rows,
                assignments_by_tag,
            )

            replacement_map: dict[str, str] = {}
            for row in target_rows:
                original_tag = str(row["original_tag"])
                pre_tag = str(row.get("replacement_tag") or original_tag)
                final_tag = final_tags_by_original.get(original_tag, pre_tag)
                if pre_tag != final_tag:
                    replacement_map[pre_tag] = final_tag
            updated_text = _apply_tag_mapping_in_text(anonymized_text, replacement_map)
        except RelationParseError as exc:
            fallback_status = "fallback_parse_error"
            fallback_error = str(exc)
        except RelationRequestError as exc:
            fallback_status = "fallback_request_error"
            fallback_error = str(exc)

    changes: list[dict[str, Any]] = []
    for row in relation_rows:
        original_tag = str(row["original_tag"])
        pre_tag = str(row.get("replacement_tag") or original_tag)

        entry: dict[str, Any] = {
            "person_index": row.get("person_index"),
            "source_text": row.get("source_text"),
            "original_tag": original_tag,
            "pre_postprocess_tag": pre_tag,
            "postprocess_tag": pre_tag,
            "changed": False,
            "reviewed_by_model": False,
            "decision_source": "skipped",
            "postprocess_status": "skipped_not_generated",
            "keep_current_tag": True,
            "standardized_tag_base": None,
            "entity_group": None,
            "rationale": "Skipped because relation tag was not model-generated patient relation output.",
        }

        if original_tag in target_tags:
            entry["reviewed_by_model"] = True

            if fallback_status:
                entry["decision_source"] = "fallback"
                entry["postprocess_status"] = fallback_status
                entry["rationale"] = (
                    "Postprocessing LLM call failed; kept relation-stage replacement tag unchanged."
                )
                if fallback_error:
                    entry["error"] = fallback_error
            else:
                assignment = assignments_by_tag[original_tag]
                state = state_by_tag.get(original_tag, {})
                post_tag = final_tags_by_original.get(original_tag, pre_tag)
                changed = post_tag != pre_tag

                entry["postprocess_tag"] = post_tag
                entry["changed"] = changed
                entry["decision_source"] = "model"
                entry["postprocess_status"] = "updated" if changed else "unchanged"
                entry["keep_current_tag"] = bool(assignment.keep_current_tag)
                entry["standardized_tag_base"] = state.get("standardized_tag_base")
                entry["entity_group"] = state.get("entity_group")
                entry["rationale"] = assignment.rationale

        changes.append(entry)

    return updated_text, changes
