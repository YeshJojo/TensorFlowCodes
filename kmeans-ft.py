import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

datapoints = np.array([[1, 1], [1, 2], [2, 2], [6, 2], [7, 2], [6, 6], [7, 6], ])

def input_fn():
    return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(datapoints, dtype=tf.float32), num_epochs=1)

num_clusters = 3
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)

num_iterations = 10
previous_centers = None
for j in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    # if previous_centers is not None:
    # print('delta:', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

centroidX = []
centroidY = []
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(datapoints):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    centroidX.append(center[0])
    centroidY.append(center[1])
    print('point:', point, 'is in cluster', cluster_index, 'centered at', center)

for centroid in cluster_centers:
    plt.scatter(centroidX, centroidY, marker="o", color="r", s=150, linewidths=5)

plt.scatter(datapoints[:, 0], datapoints[:, 1], color="b")
plt.show()