
import os
import json
import re

from pathlib import Path



def LoadReports(fp):
    with open(fp, 'r') as file:
        # Deserialize the file content into a Python dictionary
        Corpus = json.load(file)
    return Corpus

def CreateOutputDir(savedir):
    try:
        os.mkdir(savedir)
        #print(f"Directory '{savedir}' created successfully.")
    except FileExistsError:
        pass
        #print(f"Directory '{savedir}' already exists.")
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



def load_skiplist_from_directory(directory_path, initial_list=None):
    """
    Reads all .txt files in a directory and merges them with an initial list.
    
    Args:
        directory_path (str or Path): Path to the folder containing .txt files.
        initial_list (list, optional): Existing words to include.
        
    Returns:
        list: A sorted list of unique words from the directory and initial list.
    """
    # Convert to Path object and initialize set for deduplication
    base_path = Path(directory_path)
    combined_set = set(initial_list) if initial_list else set()

    # Check if directory exists
    if not base_path.is_dir():
        print(f"Warning: Directory '{directory_path}' not found.")
        return list(combined_set)

    # Iterate through all .txt files
    for file_path in base_path.glob("*.txt"):
        with file_path.open("r", encoding="utf-8") as f:
            # .strip() removes whitespace/newlines
            # if line.strip() ignores empty lines
            combined_set.update(line.strip() for line in f if line.strip())

    return sorted(list(combined_set))




class PIIFilter:
    def __init__(self, skiplist, timewords, generalwords):
        """
        Initialize with the specific lists used for filtering.
        Converting them to sets makes lookups much faster.
        """
        self.skiplist = {word.lower() for word in skiplist}
        self.timewords = set(timewords)
        self.generalwords = set(generalwords)

    def is_pii(self, text):
        """
        The main 'gatekeeper' method. 
        Returns True if the text SHOULD be treated as PII.
        Returns False if it matches any of your 'clean' criteria.
        """
        if self.check_skiplist(text):
            return False
        if self.has_timewords(text):
            return False
        if self.has_general_words(text):
            return False
        return True

    def has_timewords(self, text):
        # Splits by characters and counts occurrences of timewords
        words = re.split(r'[ -/]+', text.lower())
        return any(word in self.timewords for word in words)

    def has_general_words(self, text):
        # Standard split and count occurrences of generalwords
        return any(word in self.generalwords for word in text.lower().split())

    def check_skiplist(self, text):
        # Returns True if text is in the list
        return text.lower() in self.skiplist



