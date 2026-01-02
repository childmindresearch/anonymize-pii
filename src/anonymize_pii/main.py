
from config import report_location, anonymize_location
from helpers import CreateOutputDir, LoadReports, RunIterator



def main():

    CreateOutputDir(anonymize_location)
    Reports = LoadReports(report_location)
    RunIterator(Reports)



if __name__ == "__main__":
    main()


