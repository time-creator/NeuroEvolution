from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # TODO: automate this
    # latest results, same as in save_util use the cwd path
    data = pd.read_csv(dir_path + '\\output\\results\\results_2019-10-29_17-15-12.csv')

    data.drop(data[data.Generation < 0].index, inplace=True)

    data.plot(x='Generation', y=['Best', 'Worst', 'Average'], color=['blue', 'red', 'green'])
    plt.show()

if __name__ == '__main__':
    main()
