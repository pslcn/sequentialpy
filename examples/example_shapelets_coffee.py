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

"""
# K-means
clusters_k = 6
data_with_timesteps = np.dstack((np.arange(0, x.shape[0]) + 1, x))[0]
clusters, centroids = k_means_with_centroids(clusters_k, data_with_timesteps, x.shape[0])
for k in range(clusters_k):
  clusters_x, clusters_y = clusters[k][:, 0], clusters[k][:, 1]
  plt.scatter(clusters_x, clusters_y)
  plt.scatter(*centroids[k], label="centroid", color="red") 
plt.show()
"""

def testing_k_means():
  df = pd.read_csv(COFFEE_TRAIN_LOC)
  train_X = df.iloc[:, 1:].to_numpy()
  return shapelets.Shapelets(1, 1, 30, 4, init_with_centroids=train_X[0], load_weights=False)


if __name__ == "__main__":
  if 0:
    num_shapelets, shapelet_min_length, length_scales = 1, 25, 5

    if 0:
      shapelet_transform = testing_k_means()
    else:
      shapelet_transform = shapelets.Shapelets(num_shapelets, 1, shapelet_min_length, length_scales=length_scales, load_weights=True)

    train(shapelet_transform, epochs=5000, lr=1e-5) ; 
    # test(shapelet_transform)
    # plot_vl_shapelets(shapelet_transform, length_scales, shapelet_min_length, num_shapelets)

  else:
    print(f"Testing shapelet-transformed representation")
    num_shapelets, shapelet_min_length, length_scales = 2, 30, 1
    shapelet_transform = shapelets.Shapelets(num_shapelets, 1, shapelet_min_length, length_scales=length_scales, load_weights=True)
    # train(shapelet_transform, epochs=5000, lr=1e-3) ; exit()

    if 0:
      fig, ax = plt.subplots(1, 2)
      ax[0].plot(shapelet_transform.shapelets[0][0].detach().numpy())
      ax[1].plot(shapelet_transform.shapelets[0][1].detach().numpy())
      plt.tight_layout()
      plt.show()

    if 0:
      df = pd.read_csv(COFFEE_TRAIN_LOC)
      train_X = df.iloc[:, 1:].to_numpy()
      train_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))
      plt.suptitle("Shapelet-Transformed Data")
      plt.xlabel("$M_{1}$")
      plt.ylabel("$M_{2}$")
      for x_idx in range(train_X.shape[0]):
        M = shapelet_transform.transformed_representation(train_X[x_idx])
        print(M, train_labels[x_idx])
        plt.scatter(M[0][0], M[0][1], color="red" if train_labels[x_idx] == 0 else "blue", label=f"{train_labels[x_idx]}")
      plt.tight_layout()
      handles, labels = plt.gca().get_legend_handles_labels()
      labels = dict(zip(labels, handles))
      plt.legend(labels.values(), labels.keys())
      plt.show()

    if 1:
      df = pd.read_csv(COFFEE_TRAIN_LOC)
      train_X = df.iloc[:, 1:].to_numpy()
      train_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))

      M = np.array([shapelet_transform.transformed_representation(train_X[x_idx])[0] for x_idx in range(train_X.shape[0])])
      print(M)
      clusters_k = 2
      clusters, centroids = k_means.k_means_with_centroids(clusters_k, M, M.shape[0])

      if 0:
        clusters_k = 6
        x = train_X[0]
        data_with_timesteps = np.dstack((np.arange(0, x.shape[0]) + 1, x))[0]
        # print(data_with_timesteps.shape) ; exit()
        clusters, centroids = k_means.k_means_with_centroids(clusters_k, data_with_timesteps, x.shape[0])
        for k in range(clusters_k):
          clusters_x, clusters_y = clusters[k][:, 0], clusters[k][:, 1]
          plt.scatter(clusters_x, clusters_y)
          plt.scatter(*centroids[k], label="centroid", color="red")
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = dict(zip(labels, handles))
        plt.legend(labels.values(), labels.keys())
        plt.show()
      elif 0:
        plt.xlabel("$M_{1}$")
        plt.ylabel("$M_{2}$")
        for k in range(clusters_k):
          clusters_x, clusters_y = clusters[k][:, 0], clusters[k][:, 1]
          plt.scatter(clusters_x, clusters_y)
          plt.scatter(*centroids[k], label="centroid", color="red")
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = dict(zip(labels, handles))
        plt.legend(labels.values(), labels.keys())
        plt.show()
