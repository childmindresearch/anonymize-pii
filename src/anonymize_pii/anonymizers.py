
from config import configs, GlinerRecognizer, Entities, timewords, generalwords, anonymize_location, replacement
from helpers import PIIFilter, CreateOutputDir, SaveOutputs
from person_relations import extract_and_apply_person_relations

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
from presidio_anonymizer.operators import Operator, OperatorType

import logging
# Set the logging level for the 'stanza' logger to WARNING or ERROR
logging.getLogger('stanza').setLevel(logging.ERROR)


class InstanceCounterAnonymizer(Operator):
    """Replace the entity values with an instance counter per entity type."""

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: dict | None = None) -> str:
        if params is None:
            raise ValueError("params is required.")

        entity_type: str = params["entity_type"]
        entity_mapping: dict[str, dict[str, str]] = params["entity_mapping"]

        entity_mapping_for_type: dict[str, str] = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            new_text: str = self.REPLACING_FORMAT.format(entity_type=entity_type, index=1)
            entity_mapping[entity_type] = {}
        else:
            if text in entity_mapping_for_type:
                return entity_mapping_for_type[text]

            previous_index: int = self._get_last_index(entity_mapping_for_type)
            new_text: str = self.REPLACING_FORMAT.format(
                entity_type=entity_type, index=previous_index + 1
            )

        entity_mapping[entity_type][text] = new_text
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: dict[str, str]) -> int:
        return len(entity_mapping_for_type)

    def validate(self, params: dict | None = None) -> None:
        if params is None:
            raise ValueError("params is required.")
        if "entity_mapping" not in params:
            raise ValueError("An input dict called `entity_mapping` is required.")
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
        anonymizer.add_anonymizer(InstanceCounterAnonymizer)
        anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig(
                    "entity_counter", params={"entity_mapping": entity_mapping}
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
            doc_data['EntityMapping'] = entity_mapping
            if person_relations:
                rel_cfg = relation_config or {}
                anon_report, relation_rows = extract_and_apply_person_relations(
                    original_text=text,
                    anonymized_text=anon_report,
                    entity_mapping=entity_mapping,
                    relation_config=rel_cfg,
                )
                doc_data['PersonRelationMapping'] = relation_rows
            
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









