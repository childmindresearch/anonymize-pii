"""Headhunter-based document parsing for the anonymization pipeline.

Parses raw reports (JSON, single-column DataFrame, or multi-column DataFrame)
into normalised ``{id: text}`` dictionaries that the anonymizer consumes.
"""

import json
import re
from pathlib import Path
from typing import Any

import headhunter
import pandas as pd
from headhunter.models import ParsedBatch, ParsedText

from config import HeadhunterDataType, parsed_report_location
from helpers import CreateOutputDir, SaveOutputs


_YAML_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)


def _load_input(
    config: dict[str, Any],
) -> dict[str, str] | pd.DataFrame:
    """Load the input file described by *config*."""
    input_path = Path(config["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data_type = HeadhunterDataType(config["data_type"])

    if data_type == HeadhunterDataType.JSON:
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"JSON input must be a {{id: text}} mapping, got {type(data).__name__}"
            )
        return data

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError(
        f"Unsupported file extension '{suffix}' for DataFrame input. "
        "Use .csv or .parquet."
    )


def _resolve_expected_headings(
    config: dict[str, Any],
) -> list[str] | None:
    """Merge *headings_to_anonymize* into *expected_headings* when applicable."""
    anonymize_all: bool = config.get("anonymize_all", True)
    headings_to_anon: list[str] = config.get("headings_to_anonymize") or []

    if anonymize_all or not headings_to_anon:
        return config.get("expected_headings")

    data_type = HeadhunterDataType(config["data_type"])
    if data_type == HeadhunterDataType.MULTI_CONTENT_COLUMN_DF:
        content_cols: list[str] = config.get("content_columns") or []
        if content_cols:
            invalid = [h for h in headings_to_anon if h not in content_cols]
            if invalid:
                raise ValueError(
                    f"headings_to_anonymize entries are not valid content_columns: {invalid}. "
                    f"Available content_columns: {content_cols}"
                )
        # process_structured_df does not support expected_headings, so return None.
        return None

    configured: list[str] | None = config.get("expected_headings")
    if configured is None:
        return list(headings_to_anon)

    missing = [h for h in headings_to_anon if h not in configured]
    if missing:
        raise ValueError(
            f"headings_to_anonymize entries not found in expected_headings: {missing}"
        )
    return configured


def _process_json(
    reports: dict[str, str],
    config: dict[str, Any],
    expected_headings: list[str] | None,
) -> list[ParsedText]:
    """Run ``headhunter.process_text`` on every entry of a ``{id: text}`` dict."""
    parser_config = config.get("parser_config")
    base_metadata: dict[str, object] | None = config.get("metadata")
    match_threshold: int = config.get("match_threshold", 80)

    documents: list[ParsedText] = []
    for doc_id, text in reports.items():
        meta = dict(base_metadata) if base_metadata else {}
        meta["id"] = doc_id
        parsed = headhunter.process_text(
            text=text,
            config=parser_config,
            metadata=meta,
            expected_headings=expected_headings,
            match_threshold=match_threshold,
        )
        documents.append(parsed)
    return documents


def _process_single_column_df(
    df: pd.DataFrame,
    config: dict[str, Any],
    expected_headings: list[str] | None,
) -> ParsedBatch:
    """Run ``headhunter.process_batch_df`` on a single‑content‑column DataFrame."""
    return headhunter.process_batch_df(
        df=df,
        content_column=str(config.get("content_column")),
        id_column=config.get("id_column"),
        metadata_columns=config.get("metadata_columns"),
        config=config.get("parser_config"),
        expected_headings=expected_headings,
        match_threshold=config.get("match_threshold", 80),
    )


def _process_multi_column_df(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> ParsedBatch:
    """Run ``headhunter.process_structured_df`` on a multi‑content‑column DataFrame."""
    return headhunter.process_structured_df(
        df=df,
        id_column=config.get("id_column"),
        metadata_columns=config.get("metadata_columns"),
        content_columns=config.get("content_columns"),
    )


def _extract_heading_content(
    parsed: ParsedText,
    headings_to_anonymize: list[str],
) -> dict[str, str]:
    """Extract content belonging to specific headings (exact match on parents)."""
    df = parsed.to_dataframe()
    result: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        parents: list[str] = row["parents"]
        for heading in headings_to_anonymize:
            if heading in parents:
                result.setdefault(heading, []).append(str(row["content"]))
                break  # attribute each content row to the first matching heading

    return {heading: "\n".join(parts) for heading, parts in result.items()}


def _build_reports_dict(
    documents: list[ParsedText],
    config: dict[str, Any],
) -> dict[str, str]:
    """Convert a list of ``ParsedText`` objects to the ``{id: text}`` format.

    Behaviour varies based on config flags:

    * ``anonymize_all=True`` → full document via ``to_markdown()`` (YAML stripped).
    * ``anonymize_all=False, separate_headings_into_reports=True`` → one entry per
      matching heading keyed as ``{id}/{heading}``.
    * ``anonymize_all=False, separate_headings_into_reports=False`` → all matching
      heading content concatenated into a single ``{id: text}`` entry.
    """
    anonymize_all = bool(config.get("anonymize_all"))
    headings_to_anon: list[str] = config.get("headings_to_anonymize") or []
    separate = bool(config.get("separate_headings_into_reports"))

    reports: dict[str, str] = {}

    for doc in documents:
        doc_key = str(doc.metadata.get("id", ""))

        if anonymize_all:
            reports[doc_key] = _YAML_FRONTMATTER_RE.sub("", doc.to_markdown()).strip()
            continue

        heading_content = _extract_heading_content(doc, headings_to_anon)
        if not heading_content:
            continue

        if separate:
            for heading, text in heading_content.items():
                reports[f"{doc_key}/{heading}"] = text
        else:
            reports[doc_key] = "\n\n".join(heading_content.values())

    return reports


def parse_reports(headhunter_config: dict[str, Any]) -> dict[str, str]:
    """Parse raw input into ``{id: text}`` reports and export to disk.

    This is the single entry point called by the pipeline. It:

    1. Loads the input file (JSON / CSV / Parquet).
    2. Resolves expected headings from the config.
    3. Dispatches to the appropriate headhunter processing function.
    4. Builds an ``{id: text}`` dict compatible with the anonymizer.
    5. Exports the dict to ``data/parsed/Parsed_Reports.json``.

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
    data_type = HeadhunterDataType(headhunter_config["data_type"])
    expected_headings = _resolve_expected_headings(headhunter_config)
    raw_input = _load_input(headhunter_config)

    documents: list[ParsedText]

    if data_type == HeadhunterDataType.JSON:
        if not isinstance(raw_input, dict):
            raise ValueError("JSON data type requires a {id: text} dict input")
        documents = _process_json(raw_input, headhunter_config, expected_headings)

    elif data_type == HeadhunterDataType.SINGLE_CONTENT_COLUMN_DF:
        if not isinstance(raw_input, pd.DataFrame):
            raise ValueError("SINGLE_CONTENT_COLUMN_DF data type requires a DataFrame input")
        batch = _process_single_column_df(raw_input, headhunter_config, expected_headings)
        if batch.errors:
            error_msgs = "; ".join(
                f"row {e.get('row_index', '?')}: {e.get('error', 'unknown')}"
                for e in batch.errors
            )
            raise ValueError(f"Parsing errors in batch: {error_msgs}")
        documents = batch.documents

    elif data_type == HeadhunterDataType.MULTI_CONTENT_COLUMN_DF:
        if not isinstance(raw_input, pd.DataFrame):
            raise ValueError("MULTI_CONTENT_COLUMN_DF data type requires a DataFrame input")
        batch = _process_multi_column_df(raw_input, headhunter_config)
        if batch.errors:
            error_msgs = "; ".join(
                f"row {e.get('row_index', '?')}: {e.get('error', 'unknown')}"
                for e in batch.errors
            )
            raise ValueError(f"Parsing errors in structured batch: {error_msgs}")
        documents = batch.documents

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    if not documents:
        raise ValueError("Parsing produced zero documents. Check input data.")

    reports = _build_reports_dict(documents, headhunter_config)
    if not reports:
        raise ValueError(
            "No content matched the configured headings. "
            "Check headings_to_anonymize against the parsed document structure."
        )

    CreateOutputDir(parsed_report_location)
    output_path = parsed_report_location / "Parsed_Reports.json"
    SaveOutputs(reports, str(output_path))

    return reports
