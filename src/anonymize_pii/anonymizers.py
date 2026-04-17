
from config import configs, GlinerRecognizer, Entities, timewords, generalwords, anonymize_location, replacement
from helpers import PIIFilter, CreateOutputDir, SaveOutputs
from person_relations import extract_and_apply_person_relations, standardize_person_relation_tags

import os
import json
import csv
import re
import spacy
import textwrap
from pathlib import Path
from collections import defaultdict

from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_anonymizer.entities import ConflictResolutionStrategy
from presidio_anonymizer.operators import Operator, OperatorType

import logging

# Set the logging level for the 'stanza' logger to WARNING or ERROR
logging.getLogger('stanza').setLevel(logging.ERROR)

PLACEHOLDER_INDEX_PATTERN = re.compile(r"^<[^>]+_(\d+)>$")


class InstanceCounterAnonymizer(Operator):
    """Replace the entity values with an instance counter per entity type."""

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: dict | None = None) -> str:
        if params is None:
            raise ValueError("params is required.")

        entity_type: str = params["entity_type"]
        entity_mapping: dict[str, dict[str, str]] = params["entity_mapping"]
        entity_type_totals: dict[str, int] = params["entity_type_totals"]

        total_for_type: int = entity_type_totals.get(entity_type, 0)
        entity_mapping_for_type: dict[str, str] = entity_mapping.setdefault(entity_type, {})

        if text in entity_mapping_for_type:
            return entity_mapping_for_type[text]

        previous_index: int = self._get_last_index(entity_mapping_for_type)
        reverse_index: int = total_for_type - previous_index

        # Presidio applies operations from right to left. Use precomputed totals to
        # assign descending IDs so placeholders appear left-to-right as 1..N.
        if reverse_index > 0:
            new_index = reverse_index
        else:
            # Fallback if totals are unavailable or exhausted.
            new_index = previous_index + 1

        new_text: str = self.REPLACING_FORMAT.format(entity_type=entity_type, index=new_index)

        entity_mapping_for_type[text] = new_text
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: dict[str, str]) -> int:
        return len(entity_mapping_for_type)

    def validate(self, params: dict | None = None) -> None:
        if params is None:
            raise ValueError("params is required.")
        if "entity_mapping" not in params:
            raise ValueError("An input dict called `entity_mapping` is required.")
        if "entity_type_totals" not in params:
            raise ValueError("An input dict called `entity_type_totals` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_counter"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


def process_full_document(text, configs, warm_engines, pii_filter, mask_arg):
    # 1. Get findings from every engine
    idx_dict = {}
    for config in configs:
        name = config['name']
        scanner = EntityScanner(warm_engines[name], pii_filter, Entities)
        idx_dict[name] = scanner.scan(text, use_chunking=(name == 'GLiNER'))

    # 2. Aggregation Logic (MasterEntities)
    # This picks the highest score if multiple engines found the same word
    master_map = {}
    for engine_results in idx_dict.values():
        for ent, (etype, score) in engine_results.items():
            if ent not in master_map or score > master_map[ent][1]:
                master_map[ent] = (etype, score)

    # 3. Create the 'Deny' Map (Grouped by Entity Type)
    type_map = defaultdict(list)
    for ent, (etype, _) in master_map.items():
        type_map[etype].append(ent)
    
    idx_dict['Deny'] = dict(type_map)

    # 4. Create the 'Redact' List (Flat list of all unique words)
    if mask_arg == 'redact':
        idx_dict['Redact'] = list(master_map.keys())

    return idx_dict



class EntityScanner:
    def __init__(self, analyzer, pii_filter, entities_to_track):
        """
        :param analyzer: The Presidio AnalyzerEngine instance.
        :param pii_filter: An instance of your PIIFilter class.
        :param entities_to_track: List of entity types (e.g., PERSON, DATE).
        """
        self.analyzer = analyzer
        self.pii_filter = pii_filter
        self.entities = entities_to_track

    def scan(self, text, use_chunking=False):
        # 1. Prepare text (chunk if necessary, otherwise wrap in a list)
        items_to_scan = self._chunk_text(text) if use_chunking else [text]
        pii_dict = {}
        # 2. Extract Entities
        for chunk in items_to_scan:
            results = self.analyzer.analyze(
                text=chunk, 
                language="en", 
                entities=self.entities
            )
            
            for res in results:
                entity_text = chunk[res.start:res.end]
                
                # 3. Apply Filtering Logic (Integrated RunAnalyzer)
                if self.pii_filter.is_pii(entity_text):
                    # 4. Deduplicate and keep highest score
                    self._update_highest_score(pii_dict, entity_text, res)

        return pii_dict

    def _chunk_text(self, text, max_length=512):
        return textwrap.wrap(
            text, 
            width=max_length, 
            break_long_words=False, 
            break_on_hyphens=False
        )

    def _update_highest_score(self, pii_dict, entity_text, res):
        """Logic from your original loop to keep the most confident entity type."""
        existing_item = pii_dict.get(entity_text)
        if existing_item is None or res.score > existing_item[1]:
            pii_dict[entity_text] = (res.entity_type, res.score)


def _count_unique_entities_by_type(text, analyzer_results, anonymizer):
    """Count unique replacement texts per entity type after conflict resolution."""
    copied_results = anonymizer._copy_recognizer_results(analyzer_results)
    copied_results.sort(key=lambda x: (x.start, x.end))

    resolved_results = anonymizer._remove_conflicts_and_get_text_manipulation_data(
        copied_results,
        ConflictResolutionStrategy.MERGE_SIMILAR_OR_CONTAINED,
    )
    merged_results = anonymizer._merge_entities_with_whitespace_between(
        text,
        resolved_results,
    )

    unique_texts_by_type = defaultdict(set)
    for result in merged_results:
        unique_texts_by_type[result.entity_type].add(text[result.start:result.end])

    return {
        entity_type: len(unique_texts)
        for entity_type, unique_texts in unique_texts_by_type.items()
    }


def _placeholder_sort_key(item):
    source_text, placeholder = item
    match = PLACEHOLDER_INDEX_PATTERN.match(placeholder)
    if match:
        return (0, int(match.group(1)), source_text.lower())
    return (1, str(placeholder), source_text.lower())


def _sorted_entity_mapping_for_output(entity_mapping):
    sorted_entity_mapping = {}
    for entity_type in sorted(entity_mapping.keys()):
        mapping_for_type = entity_mapping[entity_type]
        sorted_items = sorted(mapping_for_type.items(), key=_placeholder_sort_key)
        sorted_entity_mapping[entity_type] = {
            source_text: placeholder for source_text, placeholder in sorted_items
        }
    return sorted_entity_mapping



def AnonymizeText(text, DenyList, mask_arg='entity'):

    registry = RecognizerRegistry()

    if mask_arg in {'entity', 'counter'}:
        # Append Entities to Analyzer and run
        for key, values in DenyList.items():
            if values:
                registry.add_recognizer(PatternRecognizer(supported_entity=key, deny_list=values))
        analyzer = AnalyzerEngine(registry=registry)
        results = analyzer.analyze(text=text, language="en", entities=list(DenyList.keys()))

    else:
        FullScan = PatternRecognizer(supported_entity=replacement, deny_list=DenyList)
        analyzer = AnalyzerEngine()
        analyzer.registry.add_recognizer(FullScan)
        results = analyzer.analyze(text=text, language="en",entities=[replacement])


    anonymizer = AnonymizerEngine()
    entity_mapping = {}

    if mask_arg == 'counter':
        entity_type_totals = _count_unique_entities_by_type(text, results, anonymizer)
        anonymizer.add_anonymizer(InstanceCounterAnonymizer)
        anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig(
                    "entity_counter",
                    params={
                        "entity_mapping": entity_mapping,
                        "entity_type_totals": entity_type_totals,
                    },
                )
            },
        )
    else:
        anonymized_results = anonymizer.anonymize(
            text=text, 
            analyzer_results=results,
        )
    
    return results, anonymized_results.text, entity_mapping


def RunIterator(
    Reports,
    device,
    mask_arg,
    output_arg,
    warm_engines,
    skiplist,
    person_relations,
    relation_config,
):
    # Initialize tools once (Performance boost: sets are built only once)
    pii_filter = PIIFilter(skiplist, timewords, generalwords)
    
    # Storage for 'merged' mode
    batch_iterator, batch_anonymized, batch_log = {}, {}, {}

    for idx, text in Reports.items():
        print(f"Anonymizing {idx}")

        # Detect and Process PII
        doc_data = process_full_document(text, configs, warm_engines, pii_filter, mask_arg)
        
        # Anonymize based on mask_arg
        is_redact = (mask_arg == 'redact')
        deny_list = doc_data['Redact'] if is_redact else doc_data['Deny']
        results, anon_report, entity_mapping = AnonymizeText(text, deny_list, mask_arg=mask_arg)

        if mask_arg == 'counter':
            doc_data['EntityMapping'] = _sorted_entity_mapping_for_output(entity_mapping)
            if person_relations:
                rel_cfg = relation_config or {}
                anon_report, relation_rows = extract_and_apply_person_relations(
                    original_text=text,
                    anonymized_text=anon_report,
                    entity_mapping=entity_mapping,
                    relation_config=rel_cfg,
                )
                anon_report, postprocess_changes = standardize_person_relation_tags(
                    anonymized_text=anon_report,
                    relation_rows=relation_rows,
                    relation_config=rel_cfg,
                )
                doc_data['PersonRelationMapping'] = relation_rows
                doc_data['PersonRelationPostprocessingChanges'] = postprocess_changes
            
        # Handle Output Logic
        pii_results_serialized = [result.to_dict() for result in results]

        if output_arg == 'single':
            # Save to individual subdirectories immediately
            report_path = os.path.join(anonymize_location, str(idx))
            CreateOutputDir(report_path)
            
            SaveOutputs(doc_data, f'{report_path}/Iterator.json')
            SaveOutputs(anon_report, f'{report_path}/Anonymized_Report.json')
            SaveOutputs(pii_results_serialized, f'{report_path}/PII_Log.json')
            SaveOutputs({idx: text}, f'{report_path}/Original_Report.json')
        else:
            # Store in memory for batch saving at the end
            batch_iterator[idx] = doc_data
            batch_anonymized[idx] = anon_report
            batch_log[idx] = pii_results_serialized

    # Final Batch Save (only for 'merged' or 'batch' mode)
    if output_arg != 'single':
        SaveOutputs(batch_iterator, f'{anonymize_location}/Iterator.json')
        SaveOutputs(batch_anonymized, f'{anonymize_location}/Anonymized_Reports.json')
        SaveOutputs(batch_log, f'{anonymize_location}/PII_Log.json')

    print("Anonymization Complete")









