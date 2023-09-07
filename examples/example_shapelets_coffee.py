import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sequentialpy import shapelets
from sequentialpy import k_means

COFFEE_TRAIN_LOC = "~/Downloads/UCR_TS_Archive_2015/Coffee/Coffee_TRAIN"
COFFEE_TEST_LOC = "~/Downloads/UCR_TS_Archive_2015/Coffee/Coffee_TEST"

def shuffle_dataset(train_X, train_labels):
  shuffled_idxs = np.arange(0, train_X.shape[0], dtype=int)
  np.random.shuffle(shuffled_idxs)
  return train_X[shuffled_idxs], train_labels[shuffled_idxs]

def train(shapelet_transform, epochs=1000, lr=0.01):
  df = pd.read_csv(COFFEE_TRAIN_LOC)
  train_X = df.iloc[:, 1:].to_numpy()
  train_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))
  train_X, train_labels = shuffle_dataset(train_X, train_labels)
  shapelet_transform.learn(train_X, train_labels, epochs=epochs, lr=lr)

def test(shapelet_transform):
  test_df = pd.read_csv(COFFEE_TEST_LOC)
  test_X = test_df.iloc[:, 1:].to_numpy()
  shapelet_transform.pregenerate_segment_idxs(test_X.shape[1])
  test_labels = test_df.iloc[:, 0].to_numpy().reshape((-1, 1))
  for i in range(test_X.shape[0]):
    pred = shapelet_transform.forward(test_X[i])
    print(f"[{i}] pred: {np.around(pred.detach().numpy(), 3)} actual: {test_labels[i]}") 

def testing_k_means():
  df = pd.read_csv(COFFEE_TRAIN_LOC)
  train_X = df.iloc[:, 1:].to_numpy()
  return shapelets.Shapelets(1, 1, 30, 4, init_with_centroids=train_X[0], load_weights=False)


if __name__ == "__main__":
  # num_shapelets, shapelet_min_length, length_scales = 1, 25, 5
  num_shapelets, shapelet_min_length, length_scales = 2, 30, 1
  shapelet_transform = shapelets.Shapelets(num_shapelets, 1, shapelet_min_length, length_scales=length_scales, load_weights=True)

  df = pd.read_csv(COFFEE_TEST_LOC)
  test_X = df.iloc[:, 1:].to_numpy()
  test_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))

  fig, (ax1, ax2) = plt.subplots(2, 1)

  ax1.set_title("Shapelet-Transformed Data")
  ax1.set_xlabel("$M_{1}$")
  ax1.set_ylabel("$M_{2}$")
  for x_idx in range(test_X.shape[0]):
    M = shapelet_transform.transformed_representation(test_X[x_idx])
    # print(M, test_labels[x_idx])
    ax1.scatter(M[0, 0], M[0, 1], color="red" if test_labels[x_idx] == 0 else "blue", label=f"{test_labels[x_idx]}")
  handles, labels = ax1.get_legend_handles_labels()
  labels = dict(zip(labels, handles))
  ax1.legend(labels.values(), labels.keys())

  train_X = df.iloc[:, 1:].to_numpy()
  train_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))

  # SVM
  M = np.array([shapelet_transform.transformed_representation(train_X[x_idx])[0] for x_idx in range(train_X.shape[0])])
  # M = np.array([shapelet_transform.transformed_representation(test_X[x_idx])[0] for x_idx in range(test_X.shape[0])])
  clusters_k = 2
  clusters, centroids = k_means.k_means_with_centroids(clusters_k, M, M.shape[0])
  print(centroids)
  hyperplane = np.zeros((2))
  hyperplane[1] = np.divide(*np.subtract(*centroids)[::-1])
  midpoint = np.mean(centroids, axis=0)
  print(midpoint)
  # hyperplane[1] = -(1 / hyperplane[1])
  hyperplane[1] = -(1 / hyperplane[1])
  hyperplane[0] = midpoint[1] - hyperplane[1] * midpoint[0]

  """
  support_vecs = np.zeros((2, 2))
  support_vecs[0][1], support_vecs[1][1] = hyperplane[1], hyperplane[1]

  halfway_midpoints = [np.mean([centroids[0], midpoint], axis=0), np.mean([centroids[1], midpoint], axis=0)]
  print(halfway_midpoints)
  support_vecs[0][0], support_vecs[1][0] = halfway_midpoints[0][1] - support_vecs[0][1] * halfway_midpoints[0][0], halfway_midpoints[1][1] - support_vecs[1, 1] * halfway_midpoints[1][0]
  """

  # Find closest points to hyperplane
  # Support vectors are: hyperplane[1] and close_point[1] - hyperplane[1] * close_point[0]
  # Margin is distance between both support vectors
  # Aim to maximise the margin

  # closest_cluster_points = [get_support_vecs(clusters[0], hyperplane), get_support_vecs(clusters[1], hyperplane)]

  # calc_y_intercept = lambda gradient, through_point: through_point[1] - gradient * through_point[0]

  # support_vecs = np.zeros((2, 2))
  # support_vecs[0, 1], support_vecs[1, 1] = hyperplane[1], hyperplane[1]
  # support_vecs[0, 0], support_vecs[1, 0] = calc_y_intercept(support_vecs[0, 1], clusters[0][closest_cluster_points[0]]), calc_y_intercept(support_vecs[1, 1], clusters[1][closest_cluster_points[1]])

  ax2.set_xlabel("$M_{1}$")
  ax2.set_ylabel("$M_{2}$")
  for k in range(clusters_k):
    clusters_x, clusters_y = clusters[k][:, 0], clusters[k][:, 1]
    ax2.scatter(clusters_x, clusters_y)
    ax2.scatter(*centroids[k], label="centroid", color="black")
  ax2.scatter(*midpoint)
  xx = np.linspace(0.05, 0.1)
  # ax2.scatter(*clusters[0][closest_cluster_points[0]])
  # ax2.scatter(*clusters[1][closest_cluster_points[1]])
  ax2.plot(xx, xx * hyperplane[1] + hyperplane[0], linestyle="dashed")
  ax2.plot(xx, xx * support_vecs[0, 1] + support_vecs[0, 0], color="blue")
  ax2.plot(xx, xx * support_vecs[1, 1] + support_vecs[1, 0], color="blue")
  handles, labels = ax2.get_legend_handles_labels()
  labels = dict(zip(labels, handles))
  ax2.legend(labels.values(), labels.keys())

  plt.tight_layout()
  plt.show()
