import csv
import os
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

def save_results(results):
    # TODO: Save results in a csv file in a form, so that matplotlib can easily
    # display all the given information in a good way
    # TODO: save Generation, and some other useful information in that file too

    # path and name of the new file
    new_csv_path = Path(dir_path + f"\\results\\results_{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}.csv")

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

def save_generation(generation, number):
    if not os.path.exists(Path(dir_path + f'\\generations\\gen{number}')):
        os.mkdir(Path(dir_path + f'\\generations\\gen{number}'))
    else:
        print('There already is a file with weights to the current generation! \nWeights will get overwritten!')

    for count, individual in enumerate(generation):
        weights_path = Path(dir_path + f'\\generations\\gen{number}\\weights{count}.pt')
        torch.save(individual.weight.data, weights_path)

def load_generation(gen_number, size):
    generation = []
    # 4 is the number of individuals in a generation
    for i in range(size):
        generation.append(nn.Conv2d(512, 3, kernel_size=1))

    for count, individual in enumerate(generation):
        individual.weight.data = torch.load(Path(dir_path + f'\\generations\\gen{gen_number}\\weights{count}.pt'))

    return generation
