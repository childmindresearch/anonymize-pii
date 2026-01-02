

from config import configs, gliner_recognizer, Entities, timewords, generalwords, skiplist, anonymize_location, replacement

import os
import json
import csv
import re
import spacy
import textwrap

from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine, SpacyNlpEngine, NerModelConfiguration
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig



import logging
# Set the logging level for the 'stanza' logger to WARNING or ERROR
logging.getLogger('stanza').setLevel(logging.ERROR)



##############################################################################################
# Manage filepaths, data locations
##############################################################################################

def LoadReports(fp):
    with open(fp, 'r') as file:
        # Deserialize the file content into a Python dictionary
        Corpus = json.load(file)
    return Corpus

def CreateOutputDir(savedir):
    try:
        os.mkdir(savedir)
        print(f"Directory '{savedir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{savedir}' already exists.")
    except FileNotFoundError:
        print(f"Parent directory does not exist. Use os.makedirs() to create intermediate directories.")
    except OSError as e:
        print(f"An OS error occurred: {e}")

def SaveOutputs(data, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Dictionary successfully saved to {filename}")
    except TypeError as e:
        print(f"Error: Unable to serialize data. {e}")
    except IOError as e:
        print(f"Error: Could not open or write to file. {e}")


##############################################################################################
# NLP Filtering and Cleanup Functions
##############################################################################################

# Add a time bound term skiplist
def cleantime(text):
    words = re.split(r'[ -/]+', text)
    flag=0
    for word in words:
        if word.lower() in timewords:
            flag+=1
        else:
            continue
    return flag

# Add a common term skiplist
def cleangeneral(text):
    words = text.split()
    flag=0
    for word in words:
        if word.lower() in generalwords:
            flag+=1
        else:
            continue
    return flag

def checkskiplist(text):
    if text.lower() in skiplist:
        flag=1
    else:
        flag=0
    return flag


def chunk_text_textwrap(text, max_length=384):
  """
  Splits text into chunks of a maximum character length without breaking words.
  """
  # Use textwrap.wrap to handle the logic.
  # width sets the max_length.
  # break_long_words and break_on_hyphens set to False ensure word integrity.
  chunks = textwrap.wrap(
      text,
      width=max_length,
      break_long_words=False,
      break_on_hyphens=False
  )
  return chunks

##############################################################################################
# Core Functions
##############################################################################################


# Processes one document at a time
# Returns dict with {[idx]:[{conf1...confn outputs, combined DenyList}]}
def DocLoader(text):
    idx_dict={}
    for config in configs:
        response = RunAnalyzer(text, config)
        idx_dict[config.get('name')] = response

    #Merge each config output deny list to single comprehensive DenyList (set)
    DenyList = {*()} # Start with empty
    for i, piilist in idx_dict.items():
        DenyList.update(idx_dict.get(i))
    
    idx_dict['Deny'] = list(DenyList)
    return idx_dict



# Scans through original text
# returns list of all entities identified in format [entity, type, score, pos_start, pos_end]
# returns unique dict() as Entity:(type, score) using highest score of entity type
def RunPII(text, analyzer):
    analyzer_results = analyzer.analyze(text=text, language="en", entities=Entities)
    #PII={res.start:(text[res.start:res.end],res.entity_type, res.score) for res in analyzer_results}
    PII = [(text[res.start:res.end], res.entity_type, res.score, res.start, res.end) for res in analyzer_results]
    PII_dict=dict()
    for i in PII:
        item = PII_dict.get(i[0])
        if item == None:
            PII_dict[i[0]] = (i[1],i[2])
        elif i[3] > item[1]:
            PII_dict[i[0]] = (i[1], i[2])
        else:
            pass

    return PII, PII_dict


# Copy of RUNPII - Scans through original text as Chunks
# returns list of all entities identified in format [entity, type, score, pos_start, pos_end]
# returns unique dict() as Entity:(type, score) using highest score of entity type
def RunPIIChunks(chunks, analyzer):
    PII = []
    PII_dict=dict()

    for text in chunks:
        analyzer_results = analyzer.analyze(text=text, language="en", entities=Entities)
        PII_Chunk = [(text[res.start:res.end], res.entity_type, res.score, res.start, res.end) for res in analyzer_results]
        PII.extend(PII_Chunk)

    for i in PII:
        item = PII_dict.get(i[0])
        if item == None:
            PII_dict[i[0]] = (i[1],i[2])
        elif i[3] > item[1]:
            PII_dict[i[0]] = (i[1], i[2])
        else:
            pass

    return PII, PII_dict

# This is the iterator Analyzer Engine to Flag the PII we want to mask from each document
# This function Scans the report document and creates the initial set of <MASK> entities, accounting for skiplists and addlists
def RunAnalyzer(text, config):
    if config.get('name') == 'GLiNER':
        provider = NlpEngineProvider(nlp_configuration=config.get('config'))
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        # Add the GLiNER recognizer to the registry
        analyzer.registry.add_recognizer(gliner_recognizer)
        # Remove the spaCy recognizer to avoid NER coming from spaCy
        analyzer.registry.remove_recognizer("SpacyRecognizer")

        # Chunk texts to ensure full compatibility with GLiNER NER
        chunks = chunk_text_textwrap(text)

        # Run Analyzer with given config
        PII, pdict = RunPIIChunks(chunks, analyzer)

    else:
        provider = NlpEngineProvider(nlp_configuration=config.get('config'))
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        # Run Analyzer with given config
        PII, pdict = RunPII(text, analyzer)
    
    # Clean up time bound entities - add an exclusion/drop list from the entity anonymizer (where they should not be removed)
    # Clean up General entities - add an exclusion/drop list from the entity anonymizer (where they should not be removed)
    timedrop, generaldrop, skipset = {}, {}, {}
    dropset=set()
    for k in pdict.keys():
        # Time bound
        timedrop[k]=cleantime(k)
        tdrop = [key for key, value in timedrop.items() if value >= 1]
        # General
        generaldrop[k]=cleangeneral(k)
        gdrop = [key for key, value in generaldrop.items() if value >= 1]
        # skiplist
        skipset[k]=checkskiplist(k)
        skipdrop = [key for key, value in skipset.items() if value >= 1]

    # Filter time bound and eneral entities we want to keep in the data from the masking set
    dropset = list(set(tdrop + gdrop + skipdrop))
    rdict = {key:value for key, value in pdict.items() if key not in dropset}
        
    return rdict



# Analyzer with specific Deny list and Anonymize Texts
# This function inputs the custom <MASK> entities (DenyList) and anonymizes the full text for each document
def AnonymizeText(text, DenyList):

    # Reset Analyzer and run
    FullScan = PatternRecognizer(supported_entity=replacement, deny_list=DenyList)
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(FullScan)
    results = analyzer.analyze(text=text, language="en",entities=[replacement])

    # clean up format for logging
    #results=[r.to_dict() for r in results]
    
    # Anonymizer
    anonymizer = AnonymizerEngine()
    anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )
    
    return results, anonymized_results.text


# Iterate through reports and run anonymizer functions
def RunIterator(Reports):

    PII_Iterator, Anonymized_text, PII_Log = {},{},{}

    for idx, text in Reports.items():
        print(f"Anonymizing {idx}")
        idx_dict = DocLoader(text)
        PII_Iterator[idx]=idx_dict

        DenyList=idx_dict.get('Deny')
        results, anon_report = AnonymizeText(text, DenyList)
        PII_Log[idx] = [result.to_dict() for result in results]
        Anonymized_text[idx] = anon_report

    # Save anonymized results to output directory
    SaveOutputs(PII_Iterator, f'{anonymize_location}/Iterator.json')
    SaveOutputs(Anonymized_text, f'{anonymize_location}/Anonymized_Reports.json')
    SaveOutputs(PII_Log, f'{anonymize_location}/PII_Log.json')
















