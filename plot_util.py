from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def main():
    # TODO: automate this
    # latest results
    data = pd.read_csv('D:/workFolder/NeuroEvolution/results/results_2019-06-08_14-51-22.csv')

    data.drop(data[data.Generation < 0].index, inplace=True)

    data.plot(kind='scatter', x='Generation',y='Best', color='blue')
    plt.show()

if __name__ == '__main__':
    main()
