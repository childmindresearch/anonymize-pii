

from config import configs, GlinerRecognizer, Entities, timewords, generalwords, skiplist, anonymize_location, replacement

import os
import json
import csv
import re
import spacy
import textwrap
from collections import defaultdict

from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine, SpacyNlpEngine, NerModelConfiguration
from presidio_analyzer.recognizer_registry import RecognizerRegistry
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
        #print(f"Dictionary successfully saved to {filename}")
    except TypeError as e:
        print(f"Error: Unable to serialize data. {e}")
    except IOError as e:
        print(f"Error: Could not open or write to file. {e}")


##############################################################################################
# NLP Filtering and Cleanup Functions
##############################################################################################


# Add a time bound term skiplist
def cleantime(text):
    # Splits by characters and counts occurrences of timewords
    words = re.split(r'[ -/]+', text.lower())
    return sum(1 for word in words if word in timewords)


# Add a common term skiplist
def cleangeneral(text):
    # Standard split and count occurrences of generalwords
    return sum(1 for word in text.lower().split() if word in generalwords)

def checkskiplist(text):
    # Returns 1 if text is in list, else 0
    return int(text.lower() in skiplist)


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

# Loads Documents
# Returns dict with {[idx]:[{conf1...confn outputs, combined DenyList}]}
def DocLoader(text, mask_arg):
    # Dictionary comprehension for clean initialization
    idx_dict = {config.get('name'): RunAnalyzer(text, config) for config in configs}
    
    # Merge and categorize by type

    idx_dict['Deny'] = MasterEntities(idx_dict)
    
    if mask_arg == 'redact':
        DenyList = {*()} # Start with empty
        for i, piilist in idx_dict.items():
            DenyList.update(idx_dict.get(i))    
        idx_dict['Redact'] = list(DenyList)

    return idx_dict
    


# Update Aggregator for Applying Entity Type Labels

def MasterEntities(data):
    master_dict = {}
    for sub_dict in data.values():
        for entity, (e_type, score) in sub_dict.items():
            if entity not in master_dict or score > master_dict[entity][1]:
                master_dict[entity] = (e_type, score)

    # Group by entity type using defaultdict
    type_map = defaultdict(list)
    for entity, (e_type, _) in master_dict.items():
        type_map[e_type].append(entity)
        
    return dict(type_map)




# Scans through original text
# returns list of all entities identified in format [entity, type, score, pos_start, pos_end]
# returns unique dict() as Entity:(type, score) using highest score of entity type
def RunPII(text, analyzer):
    analyzer_results = analyzer.analyze(text=text, language="en", entities=Entities)
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

    return PII_dict


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

    return PII_dict

# This is the iterator Analyzer Engine to Flag the PII we want to mask from each document
# This function Scans the report document and creates the initial set of <MASK> entities, accounting for skiplists and addlists
def RunAnalyzer(text, config):
    # Setup logic (Standardized provider/engine initialization)
    #conf_name = config.get('name')
    provider = NlpEngineProvider(nlp_configuration=config.get('config'))
    engine = provider.create_engine()

    if config.get('name') == 'GLiNER':
        # Create the registry and add GLiNER recognizer
        registry = RecognizerRegistry()
       #registry.load_predefined_recognizers() # Loads default regex-based ones
        registry.add_recognizer(GlinerRecognizer(model_name=config.get('external_model'), labels=Entities))
        analyzer = AnalyzerEngine(nlp_engine=engine, registry=registry)
        # Drop base Recognizer since it is not required for GLiNER
        #analyzer.registry.remove_recognizer("SpacyRecognizer")

        # Chunk texts to ensure full compatibility with GLiNER NER
        chunks = chunk_text_textwrap(text)
        pdict = RunPIIChunks(chunks, analyzer)

    else:
        analyzer = AnalyzerEngine(nlp_engine=engine)
        pdict = RunPII(text, analyzer)

    # Filters out entities that meet any of the "drop" criteria
    return {
        ent: val for ent, val in pdict.items() 
        if not (cleantime(ent) >= 1 or cleangeneral(ent) >= 1 or checkskiplist(ent) >= 1)
    }
    


# Analyzer with specific Deny list and Anonymize Texts
# This function inputs the custom <MASK> entities (DenyList) and anonymizes the full text for each document
def AnonymizeText(text, DenyList, entity_names=True):

    registry = RecognizerRegistry()

    if entity_names==True:
        # Append Entities to Analyzer and run
        for key, values in DenyList.items():
            recognizer = PatternRecognizer(supported_entity=key, deny_list=values)
            registry.add_recognizer(recognizer)
            analyzer = AnalyzerEngine(registry=registry)
            results = analyzer.analyze(text=text, language="en")
    else:
        FullScan = PatternRecognizer(supported_entity=replacement, deny_list=DenyList)
        analyzer = AnalyzerEngine()
        analyzer.registry.add_recognizer(FullScan)
        results = analyzer.analyze(text=text, language="en",entities=[replacement])


    anonymizer = AnonymizerEngine()
    anonymized_results = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )
    
    return results, anonymized_results.text


# Iterate through reports and run anonymizer functions
def RunIterator(Reports, mask_arg, output_arg):


    if output_arg == 'single':
        OutputIndReport(Reports, mask_arg)
        print("Anonymizer Complete")
    
    else:

        PII_Iterator, Anonymized_text, PII_Log = {},{},{}
        for idx, text in Reports.items():
            print(f"Anonymizing {idx}")
            idx_dict = DocLoader(text, mask_arg)
            PII_Iterator[idx]=idx_dict

            if mask_arg == 'redact':
                DenyList=idx_dict.get('Redact')
                results, anon_report = AnonymizeText(text, DenyList, entity_names=False)
            else:
                DenyList=idx_dict.get('Deny')
                results, anon_report = AnonymizeText(text, DenyList)
                
            PII_Log[idx] = [result.to_dict() for result in results]
            Anonymized_text[idx] = anon_report

        # Save anonymized results to output directory
        SaveOutputs(PII_Iterator, f'{anonymize_location}/Iterator.json')
        SaveOutputs(Anonymized_text, f'{anonymize_location}/Anonymized_Reports.json')
        SaveOutputs(PII_Log, f'{anonymize_location}/PII_Log.json')

        print("Anonymizer Complete")




def OutputIndReport(Reports, mask_arg):

    for idx, rawtext in Reports.items():
        PII_Iterator, Anonymized_text, PII_Log = {},{},{}
        
        print(f"Anonymizing {idx}")
        idx_dict = DocLoader(rawtext, mask_arg)
        PII_Iterator[idx]=idx_dict

        if mask_arg == 'redact':
            DenyList=idx_dict.get('Redact')
            results, anon_report = AnonymizeText(rawtext, DenyList, entity_names=False)
        else:
            DenyList=idx_dict.get('Deny')
            results, anon_report = AnonymizeText(rawtext, DenyList)
            
        PII_Log[idx] = [result.to_dict() for result in results]
        Anonymized_text[idx] = anon_report

         # Save anonymized results to output directory
        report_path = os.path.join(anonymize_location,str(idx))
        CreateOutputDir(report_path)
        SaveOutputs(PII_Iterator, f'{report_path}/Iterator.json')
        SaveOutputs(Anonymized_text, f'{report_path}/Anonymized_Report.json')
        SaveOutputs(PII_Log, f'{report_path}/PII_Log.json')

        source_text = {}
        source_text[idx] = rawtext
        SaveOutputs(source_text, f'{report_path}/Original_Report.json')











