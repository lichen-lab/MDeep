import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


def hac(cor):

    def mydist(p1, p2):
        x = int(p1)
        y = int(p2)
        return 1.0 - cor[x, y]

    x = list(range(cor.shape[0]))
    X = np.array(x)

    linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')

    result = dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    indexes = result.get('ivl')

    index =[]
    for i, itm in enumerate(indexes):
        index.append(int(itm))

    return index

if __name__ == '__main__':
    c = np.load("data/Malawiantwin_pairs/c.npy")
    hac(c)
