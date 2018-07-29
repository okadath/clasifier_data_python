#print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

import numpy as np

from sklearn.cluster import KMeans
import functions as f

a=f.lee('datos_ola.json')
print(a)
# import some data to play with
iris = datasets.load_iris()

''' le puedes aventar  asi la info :D
[[ 5.1  ,3.5 , 1.4 , 0.2],
 [ 4.9 , 3.  , 1.4 , 0.2],
 [ 4.7 , 3.2 , 1.3 , 0.2],
 [ 4.6 , 3.1 , 1.5  ,0.2]]
 '''

# X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# plt.figure(2, figsize=(8, 6))
# plt.clf()

# # Plot the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
a=np.array(
       [
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
print(iris.data)
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
y_pred = KMeans(n_clusters=5).fit_predict(iris.data)

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],c=y_pred,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Caracteristicas con Machine Learning")
ax.set_xlabel("1er eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2do eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3er eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()