from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def main():
    # TODO: automate this
    # latest results, same as in save_util use the cwd path
    data = pd.read_csv('csv_path')

    data.drop(data[data.Generation < 0].index, inplace=True)

    data.plot(x='Generation', y=['Best', 'Worst', 'Average'], color=['blue', 'red', 'green'])
    plt.show()

if __name__ == '__main__':
    main()
