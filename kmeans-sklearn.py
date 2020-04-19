import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# x1 = np.array([1, 1, 2, 6, 7, 6, 7])
# x2 = np.array([1, 2, 2, 2, 2, 6, 6])

x1 = np.array(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1, 3.7,
     4.321, 8.912])
x2 = np.array(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3, 3.1,
     1.827, 3.532])

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'm']
markers = ['o', 'v', 's']
kmeans_model = KMeans(n_clusters=3).fit(X)
print("Centroid values: \n", kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, max(x1) + 1])
    plt.ylim([0, max(x2) + 1])

plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='r')
plt.show()
