
from pathlib import Path
import os
from presidio_analyzer import EntityRecognizer, RecognizerResult
from typing import List
from gliner import GLiNER

# Directory and Filepath Locations
base_path = Path.cwd()
root_dir=two_up = base_path.parents[1]
report_in = root_dir / 'data' / 'raw'
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
GLiNER_Eng = {
    'name':'GLiNER',
    'config' : {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        },
    'external_model': "nvidia/gliner-PII"
}

configs = [spacy, stanza, GLiNER_Eng]


class GlinerRecognizer(EntityRecognizer):
    def __init__(self, model_name: str, labels: List[str], **kwargs):
        self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels
        super().__init__(supported_entities=labels, **kwargs)

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        gliner_results = self.model.predict_entities(text, self.labels, threshold=0.5)
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


##############################################################################################
# Filter and Entity Configs
##############################################################################################

Entities = ['DATE_TIME', 'LOCATION', 'ORGANIZATION', 'PHONE_NUMBER', 'US_DRIVER_LICENSE', 'NRP', 'PERSON', 'EMAIL_ADDRESS',
            'CREDIT_CARD','CRYPTO','IBAN_CODE','IP_ADDRESS','MEDICAL_LICENSE','US_BANK_NUMBER','US_ITIN','US_PASSPORT','US_SSN',
           ]


timewords = ['second','seconds','minute','minutes','hour','hours','hourly','day','days','daily','week','weeks','weekly','month','months','monthly','quarterly','year','yearly',
           'day','night','morning','evening','age','ages']

generalwords = ['DSM-5','DSM-4','K-SADS','ICD-11','zoom','YouTube','Microsoft']


skiplist = [
    'mother','mothers','father','fathers','parent','parents','brother','brothers','sister','sisters','siblings','grandparents','grandmother','grandfather','co-parents',
    'paternal','maternal','family','families','home','house','caregiver','caregivers','kid','kids','peer','peers','friend','friends','individual',
    'girls','boys','he','him','his','her','she','hers','they','teen','you','person','people','child','children','adolescent','adolescents','teenager',
    'breakfast','lunch','dinner',
    'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
    'mondays','tuesdays','wednesdays','thursdays','fridays','saturdays','sundays',
    'morning','afternoon','evening','mornings','afternoons','evenings','weekend','weekends','weekday','weekdays','weeknight','weeknights',
    'school','schools','class','classroom','classmates','teacher','teachers','student','students','principal','elemantary','cafeteria','educator','educators',
    'grade','grades','homework','playground',
    'therapy','therapist','facility','hospital','team','staff','examiner','sessions','clinician','doctor','doctors','providers','patient','patients',
    'time','now',
]

replacement = 'Redact'

