
import torch
from config import (
    report_location,
    anonymize_location,
    get_warm_engines,
    configs,
    skiplist_dir,
    person_relation_config,
)
from helpers import CreateOutputDir, LoadReports, load_skiplist_from_directory
from anonymizers import RunIterator
import argparse


def main(**kwargs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_arg = kwargs.get('mask')
    output_arg = kwargs.get('output')
    person_relations = kwargs.get('person_relations')
    
    skiplist = load_skiplist_from_directory(skiplist_dir)

    CreateOutputDir(anonymize_location)
    Reports = LoadReports(report_location)

    warm_engines = get_warm_engines(configs, device)
    RunIterator(
        Reports,
        device,
        mask_arg,
        output_arg,
        warm_engines,
        skiplist,
        person_relations,
        person_relation_config,
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type = str, default = "entity", choices=["entity", "redact", "counter"])
    parser.add_argument("--output", type = str, default = "merged")
    parser.add_argument("--person-relations", action="store_true")
    args = parser.parse_args()

    main(**vars(args))


