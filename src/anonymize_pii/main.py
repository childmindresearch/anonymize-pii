
from config import report_location, anonymize_location
from helpers import CreateOutputDir, LoadReports, RunIterator
import argparse


def main(**kwargs):

    mask_arg = kwargs.get('mask')

    CreateOutputDir(anonymize_location)
    Reports = LoadReports(report_location)
    RunIterator(Reports, mask_arg)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type = str, default = "entity")
    args = parser.parse_args()

    main(**vars(args))


