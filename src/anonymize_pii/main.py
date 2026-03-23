
import argparse
import torch

from config import report_location, anonymize_location, get_warm_engines, configs, skiplist_dir, headhunter_config 
from helpers import CreateOutputDir, LoadReports, load_skiplist_from_directory
from anonymizers import RunIterator
from parsing import parse_reports


def main(**kwargs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_arg = kwargs.get('mask')
    output_arg = kwargs.get('output')
    parse_first = kwargs.get('parse')
    skiplist = load_skiplist_from_directory(skiplist_dir)

    CreateOutputDir(anonymize_location)

    if parse_first:
        Reports = parse_reports(headhunter_config)
    else:
        Reports = LoadReports(report_location)

    warm_engines = get_warm_engines(configs, device)
    RunIterator(Reports, device, mask_arg, output_arg, warm_engines, skiplist)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type = str, default = "entity", choices=["entity", "redact", "counter"])
    parser.add_argument("--output", type = str, default = "merged")
    parser.add_argument("--parse", action="store_true", help="Parse input with headhunter before anonymization.")
    args = parser.parse_args()

    main(**vars(args))


