
from pathlib import Path
import os
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

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
GLiNER = {
    'name':'GLiNER',
    'config' : {
        "nlp_engine_name": "stanza",
        "models": [{"lang_code": "en", "model_name": "en"}]
        }
}

configs = [spacy, stanza, GLiNER]

gliner_recognizer = GLiNERRecognizer(
    model_name="urchade/gliner_multi_pii-v1",
    #entity_mapping=entity_mapping,
    flat_ner=False,
    #multi_label=True,
    #map_location="cpu",
)

##############################################################################################
# Filter and Entity Configs
##############################################################################################

Entities = ['DATE_TIME', 'LOCATION', 'ORGANIZATION', 'PHONE_NUMBER', 'US_DRIVER_LICENSE', 'NRP', 'PERSON', 'EMAIL_ADDRESS',
            'CREDIT_CARD','CRYPTO','IBAN_CODE','IP_ADDRESS','MEDICAL_LICENSE',
           'US_BANK_NUMBER','US_DRIVER_LICENSE','US_ITIN','US_PASSPORT','US_SSN',
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

