[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685)

# Anonymize-PII

This repository ingests reports containing PII and applies an iterative anonymization procedure to flag and anonymize reports.


## Features

- Base feature set uses NLP and Presidio Analyzer to flag sensitive PII content.  Base config models include Spacy, Stanza, and GLiNER
- Input documents are read as .json files (by default looks for 'Reports.json' in data/raw directory) in the format of {'PatientID': 'Full body of report text to be anonymized'}
- Process iterates through each report key, value pair using each of the default model configs [spacy, stanza, GLiNER] to generate 3 output files:
    - Anonymized_Rports.json Anonymized reports saved in same format as input document {'PatientID': 'Full body of anonymized report'}
    - Iterator.json A map of all entities identified with entity type and confidence score. Format is {'PatientID': {'config_model_name': {'PII Flagged': [type, score]},}
    - PII_Log.json A map of all text replaced with index of start/end based on source input document.  Format is {'PatientID': {'entity_type': '', 'start': '', 'end': '', 'score': '', 'analysis_explanation':'','recognition_metadata': {'recognizer_name':'','recognizer_identifier':''}, }


## To Run 

Install venv dependency requirements 

Save Input Report json document as `/data/raw/Reports.json`

You can copy the `Reports.json` file from the `/tests/` directory into `/data/raw/` to test run anonymizer

from `/src/anonymize_pii` directory, run `main.py`

Masking options:

- `--mask entity`: replaces all entities with the highest confidence entity type (default)
- `--mask redact`: overrides PII replacement with the generic `<REDACTED>` label
- `--mask counter`: adds counters to the detected entity type (e.g. `<PERSON_1>`, `<ORGANIZATION_7>`, etc.) to distinguish unique entities
    - `EXPERIMENTAL`: use `--person-relations` together with `--mask counter` to replace PERSON counters with patient-relative tags inferred by local Ollama context (e.g. `<PATIENT_MOTHER>`, `<PATIENT_COUNSELOR>`, etc.). If relation extraction is uncertain or fails, the original PERSON counter tag is preserved. Requires local Ollama installation. See `config.py` for more details.


## References

https://microsoft.github.io/presidio/

https://spacy.io/

https://stanfordnlp.github.io/stanza/

https://github.com/urchade/GLiNER



