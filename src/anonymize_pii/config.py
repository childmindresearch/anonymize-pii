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


replacement = 'Redact'

