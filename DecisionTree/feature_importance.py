from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = '/Users/somakisery/Code/PyCharmProjects/venv/celebA_df.pickle'

df = pd.read_pickle(DATA_PATH)
df = df.drop(columns='target')

face_vectors = []
for i in range(10000):
    face_vectors.append(df.iloc[i])
face_vectors = np.array(face_vectors)

if not os.path.exists('./PCA'):
    os.makedirs('./PCA')


def plot_pca(n_comp):

    x_values = []
    for j in range(n_comp + 1):
        if j == 0:
            x_values.append(1)
        elif j == 1:
            continue
        else:
            x_values.append(j)

    pca = PCA(n_components=n_comp)
    pca.fit(face_vectors)

    plt.ylabel('Variance ratio', fontsize=16)
    plt.xlabel('Components', fontsize=16)
    if n_comp < 20:
        plt.xticks(range(1, n_comp + 1, 1))
    else:
        plt.xticks(range(1, n_comp + 1, 5))
    plt.plot(x_values, pca.explained_variance_ratio_)
    plt.savefig('./PCA/' + str(n_comp) + '_comp_pca.jpg')
    plt.show()


for i in range(1, 50, 5):
    plot_pca(i)
