import torch
from pathlib import Path
import os
from presidio_analyzer import EntityRecognizer, RecognizerResult,AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine, SpacyNlpEngine, NerModelConfiguration
from presidio_analyzer.recognizer_registry import RecognizerRegistry

from typing import List
from gliner import GLiNER as glmodel

# Directory and Filepath Locations
base_path = Path.cwd()
root_dir=two_up = base_path.parents[1]
report_in = root_dir / 'data' / 'raw'
skiplist_dir = root_dir / 'data' / 'external'
report_location = f'{report_in}/Reports.json'
anonymize_location = root_dir / 'data' / 'exports'


##############################################################################################
# Core Model Configs
##############################################################################################

# Load NLP model configs
spacy = {
    'name':'spacy',
    'config' : {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            ],
    }
}
stanza = {
    'name':'stanza',
    'config' : {
        "nlp_engine_name": "stanza",
        "models": [{"lang_code": "en", "model_name": "en"}]
        }
}
GLiNER = {
    'name':'GLiNER',
    'config' : {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]},
    'external_model': "nvidia/gliner-pii"
}

configs = [spacy, stanza, GLiNER]


class GlinerRecognizer(EntityRecognizer):
    def __init__(self, model_name: str, labels: List[str], device: str, **kwargs):
        self.model = glmodel.from_pretrained(model_name).to(device)
        self.labels = labels
        super().__init__(supported_entities=labels, **kwargs)

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        gliner_results = self.model.predict_entities(text, self.labels, threshold=0.5, max_len=512)
        for res in gliner_results:
            # Convert GLiNER output to Presidio RecognizerResult
            results.append(
                RecognizerResult(
                    entity_type=res["label"],
                    start=res["start"],
                    end=res["end"],
                    score=res["score"]
                )
            )
        return results

    def shutdown(self):
        # Move model to CPU and clear cache to free VRAM
        if hasattr(self, 'model'):
            self.model.to("cpu")
            del self.model
            torch.cuda.empty_cache()



def get_warm_engines(configs, device):
    warm_engines = {}
    for config in configs:
        name = config.get('name')
        
        # Pass the config dict directly to the constructor
        # Note: config.get('config') is the dictionary defined in your 'spacy' or 'stanza' variables
        provider = NlpEngineProvider(nlp_configuration=config.get('config'))
        engine = provider.create_engine()
        registry = RecognizerRegistry()

        if name == 'GLiNER':
            gliner_rec = GlinerRecognizer(
                model_name=config.get('external_model'), 
                labels=Entities, 
                device=device
            )
            registry.add_recognizer(gliner_rec)
            warm_engines[name] = AnalyzerEngine(nlp_engine=engine, registry=registry)
        else:
            # Load default recognizers (regex, etc) for SpaCy/Stanza
            registry.load_predefined_recognizers()
            warm_engines[name] = AnalyzerEngine(nlp_engine=engine, registry=registry)
            
    return warm_engines



##############################################################################################
# Filter and Entity Configs
##############################################################################################

Entities = ['DATE_TIME', 'LOCATION', 'ORGANIZATION', 'PHONE_NUMBER', 'US_DRIVER_LICENSE', 'NRP', 'PERSON', 'EMAIL_ADDRESS',
            'CREDIT_CARD','CRYPTO','IBAN_CODE','IP_ADDRESS','MEDICAL_LICENSE','US_BANK_NUMBER','US_ITIN','US_PASSPORT','US_SSN',
           ]


timewords = ['second','seconds','minute','minutes','hour','hours','hourly','day','days','daily','week','weeks','weekly','month','months','monthly','quarterly','year','yearly',
           'day','night','morning','evening','age','ages']

generalwords = ['DSM-5','DSM-4','K-SADS','ICD-11','zoom','YouTube','Microsoft']


replacement = 'REDACTED'


person_relation_config = {
    'enabled_for_mask': 'counter',
    'batch_prompt_template': (
        'Task: classify each listed person mention in a clinical note.\n'
        'Output: one JSON object only, no markdown, no prose, no code fences.\n'
        'Schema: {"assignments":[{"person_tag":"<PERSON_2>","relation_label":"mother","related_to_patient":true,"confidence":0.92,"rationale_short":"listed in parent contact"}]}\n'
        'Rules:\n'
        '1) Return exactly one assignment for every provided person_tag.\n'
        '2) Every assignment must include only these keys: person_tag, relation_label, related_to_patient, confidence, rationale_short.\n'
        '3) person_tag must exactly match one tag from the provided targets.\n'
        '4) relation_label must be short lowercase text. Use relation_label=patient when the person is the patient.\n'
        '5) Use related_to_patient=true for meaningful patient-context connections (family, caregivers, household, teachers, classmates, peers, clinicians, counselors, coaches, mentors, supervisors, recurring social contacts).\n'
        '6) Use related_to_patient=false and relation_label=unrelated only for clearly external references with no direct patient relationship (e.g., book authors, public figures, celebrities).\n'
        '7) If ambiguous between related and unrelated, prefer related_to_patient=true with your best relation_label.\n'
        '8) confidence must be a float in [0,1].\n'
        '9) rationale_short must be <= 10 words.\n'
        'Targets JSON: [[TARGETS_JSON]]'
    ),
    'ollama': {
        'url': 'http://localhost:11434/api/generate',
        'model': 'qwen3.5:27b',
        'timeout_seconds': 30,
        'temperature': 0.0,
    },
    'context_window_chars': 220,
    'confidence_threshold': 0.5,
    'max_persons_per_report': 50,
    'batch_size': 5,
}

