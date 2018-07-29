import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
# iris = datasets.load_iris()
# print (iris)

import numpy as np
import sklearn.datasets

examples = []
examples.append('some text')
examples.append('another example text')
examples.append('example 3')

target = np.zeros((3,), dtype=np.int64)
target[0] = 0
target[1] = 1
target[2] = 0
dataset = sklearn.datasets.base.Bunch(data=examples, target=target)
print(dataset)