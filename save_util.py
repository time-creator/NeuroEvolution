import csv
import os
from datetime import datetime
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

# path to the new results file
new_csv_path = Path(dir_path + f"\\results\\results_{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}.csv")

def save_results(results):
    # TODO: Save results in a csv file in a form, so that matplotlib can easily
    # display all the given information in a good way
    # TODO: save Generation, and some other useful information in that file too

    with open(new_csv_path, 'w') as new_results:
        csv_writer = csv.writer(new_results, delimiter=',')

        # first line equals column names
        header = [i for i in range(len(results[0]))]
        header[0] = 'Generation'
        header[1] = 'Best'
        header[-2] = 'Worst'
        header[-1] = 'Average'
        csv_writer.writerow(header)

        # writerows iterates over results, since it's a list / iterable
        csv_writer.writerows(results)
