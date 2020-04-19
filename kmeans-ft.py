import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x1 = np.array(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1, 3.7,
     4.321, 8.912])
x2 = np.array(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3, 3.1,
     1.827, 3.532])

# datalist = np.array([[1, 1], [1, 2], [2, 2], [6, 2], [7, 2], [6, 6], [7, 6], ])

datalist = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
plt.scatter(datalist[:, 0], datalist[:, 1], color="k")
plt.xlim([0, max(x1) + 1])
plt.ylim([0, max(x2) + 1])
plt.show()


def input_fn():
    return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(datalist, dtype=tf.float32), num_epochs=1)


num_clusters = 3
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

num_iterations = 10
previous_centers = None
for j in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    print('cluster centers - ', j, ':\n', cluster_centers)
    # if previous_centers is not None:
    #    print('cluster_centers - previous_centers:\n', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print('score:', kmeans.score(input_fn))
print('Centroid values:\n', cluster_centers)

centroidX = []
centroidY = []
cluster0 = []
cluster1 = []
cluster2 = []
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(datalist):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    centroidX.append(center[0])
    centroidY.append(center[1])
    if cluster_index == 0:
        cluster0.append(list(point))
    if cluster_index == 1:
        cluster1.append(list(point))
    if cluster_index == 2:
        cluster2.append(list(point))

print("\nPoints in Cluster 1", cluster0)
print("Points in Cluster 2", cluster1)
print("Points in Cluster 3", cluster2)
x1, x2 = zip(*cluster0)
y1, y2 = zip(*cluster1)
z1, z2 = zip(*cluster2)
for centroid in cluster_centers:
    plt.scatter(centroidX, centroidY, marker="x", color="r")

for i in range(len(cluster0)):
    plt.scatter(x1[i], x2[i], marker="o", color="g")
for j in range(len(cluster1)):
    plt.scatter(y1[j], y2[j], marker="v", color="m")
for k in range(len(cluster2)):
    plt.scatter(z1[k], z2[k], marker="s", color="b")
# plt.scatter(datalist[:, 0], datalist[:, 1], color="b")
plt.title("Kmeans Clustering")
plt.xlim([0, max(x1) + 1])
plt.ylim([0, max(x2) + 1])
plt.show()
