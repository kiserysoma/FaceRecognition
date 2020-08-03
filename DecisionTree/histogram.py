import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_pickle('/Users/somakisery/Code/PyCharmProjects/venv/celebA_df.pickle')

if not os.path.exists('./Histograms'):
    os.makedirs('./Histograms')

for i in df.columns:
    data = df[i]
    counts, bins = np.histogram(data)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Feature value', fontsize=16)
    plt.savefig('./Histograms/' + i + '_histogram.jpg')
    plt.show()
