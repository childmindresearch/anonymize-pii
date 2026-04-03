"""Headhunter-based document parsing for the anonymization pipeline.

Parses raw reports (JSON, single-column DataFrame, or multi-column DataFrame)
into normalized ``{id: text}`` dictionaries that the anonymizer consumes.
"""

import json
import re
from pathlib import Path
from typing import Literal, cast

import headhunter
import pandas as pd
from headhunter.models import ParsedBatch, ParsedText
from rapidfuzz import fuzz

from config import parsed_report_location
from helpers import CreateOutputDir, SaveOutputs


_YAML_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)
_TABLE_SUFFIXES = {".csv", ".parquet", ".pq"}
ProcessingMode = Literal["json", "single", "multi"]


def _normalize_heading(text: str) -> str:
    """Normalize heading text for case-insensitive matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _dedupe_headings_case_insensitive(headings: list[str]) -> list[str]:
    """Return headings with case-insensitive duplicates removed, preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()

    for heading in headings:
        normalized = _normalize_heading(heading)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(heading)

    return deduped


def _normalize_content_columns(config: dict) -> list[str]:
    """Return validated content column names from config."""
    raw = config.get("content_columns")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("content_columns must be a list of non-empty strings")

    columns: list[str] = []
    for col in raw:
        if not isinstance(col, str) or not col.strip():
            raise ValueError("content_columns must be a list of non-empty strings")
        columns.append(col.strip())

    return columns


def _infer_processing_mode(config: dict) -> tuple[ProcessingMode, list[str]]:
    """Infer parser mode from file extension and ``content_columns`` length."""
    suffix = Path(config["input_path"]).suffix.lower()

    if suffix == ".json":
        return "json", []

    if suffix in _TABLE_SUFFIXES:
        content_columns = _normalize_content_columns(config)
        if not content_columns:
            raise ValueError(
                "content_columns is required for CSV/Parquet input and must include at least one column"
            )
        return ("single" if len(content_columns) == 1 else "multi"), content_columns

    raise ValueError(
        f"Unsupported file extension '{suffix}'. Supported extensions are .json, .csv, .parquet, .pq"
    )


def _load_input(
    config: dict,
) -> dict[str, str] | pd.DataFrame:
    """Load the input file described by *config*."""
    input_path = Path(config["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"JSON input must be a {{id: text}} mapping, got {type(data).__name__}"
            )
        return data
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError(
        f"Unsupported file extension '{suffix}'. Supported extensions are .json, .csv, .parquet, .pq"
    )


def _resolve_expected_headings(
    config: dict,
    mode: ProcessingMode,
    content_columns: list[str],
) -> list[str] | None:
    """Merge *headings_to_anonymize* into *expected_headings* when applicable."""
    headings_to_anon: list[str] = config.get("headings_to_anonymize") or []

    if mode == "multi":
        if headings_to_anon:
            invalid = [h for h in headings_to_anon if h not in content_columns]
            if invalid:
                raise ValueError(
                    f"headings_to_anonymize entries are not valid content_columns: {invalid}. "
                    f"Available content_columns: {content_columns}"
                )
        # process_structured_df does not support expected_headings.
        return None

    if not headings_to_anon:
        return config.get("expected_headings")

    configured: list[str] | None = config.get("expected_headings")
    if configured is None:
        return _dedupe_headings_case_insensitive(list(headings_to_anon))

    return _dedupe_headings_case_insensitive([*configured, *headings_to_anon])


def _process_json(
    reports: dict[str, str],
    config: dict,
    expected_headings: list[str] | None,
) -> list[ParsedText]:
    """Run ``headhunter.process_text`` on every entry of a ``{id: text}`` dict."""
    parser_config = config.get("parser_config")
    match_threshold = int(config.get("match_threshold", 80))

    documents: list[ParsedText] = []
    for doc_id, text in reports.items():
        parsed = headhunter.process_text(
            text=text,
            config=parser_config,
            metadata={"id": doc_id},
            expected_headings=expected_headings,
            match_threshold=match_threshold,
        )
        documents.append(parsed)
    return documents


def _process_single_column_df(
    df: pd.DataFrame,
    config: dict,
    content_columns: list[str],
    expected_headings: list[str] | None,
) -> ParsedBatch:
    """Run ``headhunter.process_batch_df`` on a single‑content‑column DataFrame."""
    return headhunter.process_batch_df(
        df=df,
        content_column=content_columns[0],
        id_column=config.get("id_column"),
        config=config.get("parser_config"),
        expected_headings=expected_headings,
        match_threshold=int(config.get("match_threshold", 80)),
    )


def _process_multi_column_df(
    df: pd.DataFrame,
    config: dict,
    content_columns: list[str],
) -> ParsedBatch:
    """Run ``headhunter.process_structured_df`` on a multi‑content‑column DataFrame."""
    return headhunter.process_structured_df(
        df=df,
        id_column=config.get("id_column"),
        content_columns=content_columns,
    )


def _resolve_doc_heading_targets(
    parsed: ParsedText,
    headings_to_anonymize: list[str],
    match_threshold: int,
    allow_fuzzy_fallback: bool = True,
) -> set[str]:
    """Resolve heading targets for a document, including fuzzy-matched aliases."""
    targets = {
        _normalize_heading(heading)
        for heading in headings_to_anonymize
        if _normalize_heading(heading)
    }
    if not targets:
        return set()

    matched_headings = parsed.metadata.get("matched_headings")
    if not isinstance(matched_headings, list):
        return targets

    for match in matched_headings:
        if not isinstance(match, dict):
            continue
        match_dict = cast(dict, match)

        expected = match_dict.get("expected")
        matched_text = match_dict.get("matched_text")
        if not isinstance(expected, str) or not isinstance(matched_text, str):
            continue

        expected_norm = _normalize_heading(expected)
        if expected_norm in targets:
            matched_norm = _normalize_heading(matched_text)
            if matched_norm:
                targets.add(matched_norm)

    if allow_fuzzy_fallback:
        # Fallback: fuzzy-map unresolved targets to existing heading tokens in this document.
        doc_heading_norms = {
            _normalize_heading(ctx.token.content)
            for ctx in parsed.hierarchy
            if ctx.token.type == "heading"
        }
        doc_heading_norms.discard("")

        for target in list(targets):
            if target in doc_heading_norms:
                continue

            best_heading: str | None = None
            best_score = 0.0
            for heading_norm in doc_heading_norms:
                score = float(fuzz.ratio(target, heading_norm))
                if score > best_score:
                    best_score = score
                    best_heading = heading_norm

            if best_heading is not None and best_score >= float(match_threshold):
                targets.add(best_heading)

    return targets


def _find_heading_subtree_spans(
    parsed: ParsedText,
    target_headings: set[str],
) -> list[tuple[int, int, str]]:
    """Find hierarchy spans for heading subtrees selected by heading text."""
    if not target_headings:
        return []

    hierarchy = parsed.hierarchy
    spans: list[tuple[int, int, str]] = []

    for idx, ctx in enumerate(hierarchy):
        token = ctx.token
        if token.type != "heading":
            continue
        if _normalize_heading(token.content) not in target_headings:
            continue

        end = idx + 1
        while end < len(hierarchy):
            next_ctx = hierarchy[end]
            if next_ctx.token.type == "heading" and next_ctx.level <= ctx.level:
                break
            end += 1

        spans.append((idx, end, token.content))

    return spans


def _merge_overlapping_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping hierarchy spans while preserving source order."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda span: span[0])
    merged: list[tuple[int, int]] = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _hierarchy_to_markdown(hierarchy: list) -> str:
    """Render hierarchy contexts to markdown using headhunter-compatible rules."""
    if not hierarchy:
        return ""

    lines: list[str] = []
    i = 0

    while i < len(hierarchy):
        ctx = hierarchy[i]
        token = ctx.token

        if token.type == "heading":
            is_inline = bool(token.metadata and token.metadata.is_inline)
            has_next = i + 1 < len(hierarchy)
            next_is_content = (
                has_next
                and hierarchy[i + 1].token.type == "content"
                and hierarchy[i + 1].level == ctx.level + 1
            )

            if is_inline and next_is_content:
                lines.append(f"**{token.content}:** {hierarchy[i + 1].token.content}")
                lines.append("")
                i += 2
            elif is_inline:
                lines.append(f"**{token.content}:**")
                lines.append("")
                i += 1
            else:
                hash_count = min(ctx.level, 6)
                lines.append(f"{'#' * hash_count} {token.content}")
                lines.append("")
                i += 1
        else:
            lines.append(token.content)
            lines.append("")
            i += 1

    return "\n".join(lines).rstrip()


def _build_missing_heading_diagnostics(
    documents: list[ParsedText],
    configured_targets: list[str],
) -> str:
    """Build diagnostic text to help debug missing heading filters."""
    diagnostics: list[str] = []

    if configured_targets:
        diagnostics.append(f"headings_to_anonymize={configured_targets}")

    missing_by_doc: dict[str, list[str]] = {}
    for doc in documents:
        missing = doc.metadata.get("missing_headings")
        if isinstance(missing, list) and missing:
            missing_by_doc[str(doc.metadata.get("id", ""))] = [str(x) for x in missing]

    if missing_by_doc:
        diagnostics.append(f"missing_headings_by_doc={missing_by_doc}")

    if not diagnostics:
        return ""

    return " Diagnostics: " + "; ".join(diagnostics)


def _build_reports_dict(
    documents: list[ParsedText],
    config: dict,
    mode: ProcessingMode,
) -> dict[str, str]:
    """Convert a list of ``ParsedText`` objects to the ``{id: text}`` format.

    Behavior varies based on config flags:

        * Empty/missing ``headings_to_anonymize`` → full document via ``to_markdown()``
            (YAML stripped).
        * Non-empty ``headings_to_anonymize`` with ``separate_headings_into_reports=True``
            → one entry per matching heading subtree keyed as ``{id}/{heading}``
            (duplicate headings in a document use suffixes like ``#2``).
        * Non-empty ``headings_to_anonymize`` with ``separate_headings_into_reports=False``
            → all matching heading subtrees merged in source order into one ``{id: markdown}``.
    """
    headings_to_anon: list[str] = config.get("headings_to_anonymize") or []
    separate = bool(config.get("separate_headings_into_reports"))
    filter_mode = bool(headings_to_anon)
    use_threshold = mode in {"json", "single"}
    match_threshold = int(config.get("match_threshold", 80)) if use_threshold else 0

    reports: dict[str, str] = {}

    for doc in documents:
        doc_key = str(doc.metadata.get("id", ""))

        if not filter_mode:
            reports[doc_key] = _YAML_FRONTMATTER_RE.sub("", doc.to_markdown()).strip()
            continue

        target_headings = _resolve_doc_heading_targets(
            doc,
            headings_to_anon,
            match_threshold,
            allow_fuzzy_fallback=use_threshold,
        )
        selected_spans = _find_heading_subtree_spans(doc, target_headings)
        if not selected_spans:
            continue

        if separate:
            key_counts: dict[str, int] = {}
            for start, end, heading in selected_spans:
                section_markdown = _hierarchy_to_markdown(doc.hierarchy[start:end]).strip()
                if not section_markdown:
                    continue

                base_key = f"{doc_key}/{heading}"
                key_counts[base_key] = key_counts.get(base_key, 0) + 1
                suffix = "" if key_counts[base_key] == 1 else f"#{key_counts[base_key]}"
                reports[f"{base_key}{suffix}"] = section_markdown
        else:
            merged_spans = _merge_overlapping_spans(
                [(start, end) for start, end, _ in selected_spans]
            )

            filtered_hierarchy: list = []
            for start, end in merged_spans:
                filtered_hierarchy.extend(doc.hierarchy[start:end])

            merged_markdown = _hierarchy_to_markdown(filtered_hierarchy).strip()
            if merged_markdown:
                reports[doc_key] = merged_markdown

    return reports


def parse_reports(headhunter_config: dict) -> dict[str, str]:
    """Parse raw input into ``{id: text}`` reports and export to disk.

    This is the single entry point called by the pipeline. It:

    1. Loads the input file (JSON / CSV / Parquet).
    2. Infers mode from extension and ``content_columns`` length.
    3. Resolves expected headings from the config.
    4. Dispatches to the appropriate headhunter processing function.
    5. Builds an ``{id: text}`` dict compatible with the anonymizer.
    6. Exports the dict to ``data/parsed/Parsed_Reports.json``.

    Returns
    -------
    dict[str, str]
        The ``{id: text}`` mapping ready for anonymization.

    Raises
    ------
    ValueError
        On invalid configuration or empty results.
    FileNotFoundError
        If the input file is missing.
    """
    mode, content_columns = _infer_processing_mode(headhunter_config)
    expected_headings = _resolve_expected_headings(
        headhunter_config,
        mode,
        content_columns,
    )
    raw_input = _load_input(headhunter_config)

    documents: list[ParsedText]

    if mode == "json":
        if not isinstance(raw_input, dict):
            raise ValueError("JSON input requires a {id: text} dict")
        documents = _process_json(
            cast(dict[str, str], raw_input),
            headhunter_config,
            expected_headings,
        )

    elif mode == "single":
        if not isinstance(raw_input, pd.DataFrame):
            raise ValueError("Single-column mode requires a DataFrame input")
        batch = _process_single_column_df(
            raw_input,
            headhunter_config,
            content_columns,
            expected_headings,
        )
        if batch.errors:
            error_msgs = "; ".join(
                f"row {e.get('row_index', '?')}: {e.get('error', 'unknown')}"
                for e in batch.errors
            )
            raise ValueError(f"Parsing errors in batch: {error_msgs}")
        documents = batch.documents

    elif mode == "multi":
        if not isinstance(raw_input, pd.DataFrame):
            raise ValueError("Multi-column mode requires a DataFrame input")
        batch = _process_multi_column_df(raw_input, headhunter_config, content_columns)
        if batch.errors:
            error_msgs = "; ".join(
                f"row {e.get('row_index', '?')}: {e.get('error', 'unknown')}"
                for e in batch.errors
            )
            raise ValueError(f"Parsing errors in structured batch: {error_msgs}")
        documents = batch.documents

    else:
        raise ValueError(f"Unknown parse mode: {mode}")

    if not documents:
        raise ValueError("Parsing produced zero documents. Check input data.")

    reports = _build_reports_dict(documents, headhunter_config, mode)
    if not reports:
        diagnostics = _build_missing_heading_diagnostics(
            documents,
            headhunter_config.get("headings_to_anonymize") or [],
        )
        raise ValueError(
            "No content matched the configured headings. "
            "Check headings_to_anonymize against the parsed document structure."
            f"{diagnostics}"
        )

    CreateOutputDir(parsed_report_location)
    output_path = parsed_report_location / "Parsed_Reports.json"
    SaveOutputs(reports, str(output_path))

    return reports
