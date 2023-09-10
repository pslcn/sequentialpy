import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numba as nb

from sequentialpy import k_means

save_folder_path = "weights/"
def save_info_appendix(num_shapelets, num_categories, shapelet_min_length, length_scales): return f"{num_shapelets}_{num_categories}_{shapelet_min_length}_{length_scales}.pt"

@nb.njit(inline="always")
def generate_segment_idxs(shapelet_length, nelems): return np.arange(0, shapelet_length, dtype=np.int32) + np.arange(0, nelems - shapelet_length).reshape((-1, 1))
@nb.njit(inline="always")
def generate_vl_segment_idxs(r, nelems, shapelet_min_length): return generate_segment_idxs(r * shapelet_min_length, nelems)

@nb.njit
def generate_h_segment_idxs(length_scales, nelems, shapelet_min_length):
  h_segment_idxs = np.zeros((length_scales, nelems - shapelet_min_length, length_scales * shapelet_min_length))
  for r in range(length_scales):
    shapelet_size = (r + 1) * shapelet_min_length
    h_segment_idxs[r][:nelems - shapelet_size, :shapelet_size] = generate_vl_segment_idxs(r, nelems, shapelet_min_length)
  return h_segment_idxs

@nb.njit
def centroid_for_shapelets(num_cateories, init_with_centroids):
  # disregard timesteps (x-axis)
  centroid = k_means.k_means_with_centroids(num_categories, np.dstack((np.arange(0, init_with_centroids.shape[0]) + 1, init_with_centroids))[0], init_with_centroids.shape[0])[1]
  return centroid.reshape((2))[1]


@nb.njit(parallel=True)
def numba_mean_3d(x):
  arr_mean = np.zeros((x.shape[:2]))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      arr_mean[i][j] = np.mean(x[i, j])
  return arr_mean

@nb.njit(inline="always")
def shapelet_dists(series_segments, shapelets):
  # shape of dists: (number of shapelets, number of segments)
  return numba_mean_3d((series_segments - shapelets) ** 2).T

@nb.njit(parallel=True)
def min_shapelet_dists(dists):
  return np.array([np.min(dists[i]) for i in range(dists.shape[0])])

# arrays are made homogenous for JIT by using vector of array sizes (`range(self.length_scales)`)
@nb.njit()
def shapelet_transformed_representation(series, shapelets_as_homogenous, num_shapelets, shapelet_min_length, length_scales):
  min_dists = np.zeros((length_scales, num_shapelets))
  for r in range(length_scales):
    # shape of segment_idxs at r: (number of segments, shapelet size)
    shapelet_r_size = (r + 1) * shapelet_min_length
    num_segments = series.shape[0] - shapelet_r_size
    segment_idxs = np.arange(0, shapelet_r_size, dtype=np.int32) + np.arange(0, num_segments).reshape((-1, 1))

    series_as_segments = np.zeros((num_segments, num_shapelets, shapelet_r_size))
    for segment_idx in range(num_segments):
      series_as_segments[segment_idx][:] = series[segment_idxs[segment_idx]]

    min_dists[r] = min_shapelet_dists(shapelet_dists(series_as_segments, shapelets_as_homogenous[r][:, :shapelet_r_size]))
  return min_dists


# supports both uniform and variable shapelet lengths
# Shapelets(1, 2, 20) -> Shapelets with uniform length of 20
# Shapelets(1, 2, 20, 4) -> Shapelets with lengths (20, 40, 60, 80)
class Shapelets:
  def __init__(self, num_shapelets, num_categories, shapelet_min_length, length_scales=1, lambda_w=1e-2, init_with_centroids=None, load_weights=True):
    self.num_shapelets = num_shapelets
    self.num_categories = num_categories
    self.shapelet_min_length = shapelet_min_length
    self.length_scales = length_scales
    # regularisation parameter
    self.lambda_w = lambda_w

    info_appendix = save_info_appendix(self.num_shapelets, self.num_categories, self.shapelet_min_length, self.length_scales)
    self.save_shapelets_loc = save_folder_path + "shapelets_" + info_appendix
    self.save_biases_loc = save_folder_path + "biases_" + info_appendix
    self.save_weights_loc = save_folder_path + "weights_" + info_appendix
    if load_weights and all([os.path.exists(f) for f in [self.save_shapelets_loc, self.save_biases_loc, self.save_weights_loc]]):
      print(f"Loading weights for (num_shapelets: {self.num_shapelets}, num_categories: {self.num_categories}, shapelet_min_length: {self.shapelet_min_length}, length_scales: {self.length_scales})")
      self.shapelets = list(torch.load(self.save_shapelets_loc))
      self.biases = torch.load(self.save_biases_loc)
      self.weights = list(torch.load(self.save_weights_loc))
    else:
      self.biases = torch.rand((self.num_categories), requires_grad=True, dtype=torch.float64)
      self.shapelets, self.weights = [], []

      def shapelet_append_value(r): return torch.rand((self.num_shapelets, r * self.shapelet_min_length), requires_grad=True, dtype=torch.float64)
      if type(init_with_centroids) is np.ndarray:
        print("Initialising shapelets using K-means centroids")
        centroid = centroid_for_shapelets(self.num_categories, init_with_centroids)
        def shapelet_append_value(r): return torch.full((self.num_shapelets, r * self.shapelet_min_length), centroid, requires_grad=True, dtype=torch.float64)
      else: print("Initialising shapelets regularly")
      for r in range(1, self.length_scales + 1):
        self.weights.append(torch.rand((self.num_categories, self.num_shapelets), requires_grad=True, dtype=torch.float64))
        self.shapelets.append(shapelet_append_value(r))

  def pregenerate_segment_idxs(self, nelems: int) -> None:
    self.segment_idxs = [generate_vl_segment_idxs(r, nelems, self.shapelet_min_length) for r in range(1, self.length_scales + 1)]

  def shapelets_as_homogenous(self):
    h_shapelets = np.zeros((self.length_scales, self.num_shapelets, self.length_scales * self.shapelet_min_length))
    for r in range(self.length_scales):
      h_shapelets[r][:, :(r + 1) * self.shapelet_min_length] = self.shapelets[r]
    return h_shapelets

  def transformed_representation(self, series):
    with torch.no_grad():
      return shapelet_transformed_representation(series, self.shapelets_as_homogenous(), self.num_shapelets, self.shapelet_min_length, self.length_scales)

  def forward(self, x):
    min_dists_x_weights = 0
    for r in range(1, self.length_scales + 1):
      series_segments = torch.tensor(x[self.segment_idxs[r - 1]][:, np.newaxis].repeat(self.num_shapelets, axis=1))
      dists = torch.mean((series_segments - self.shapelets[r - 1]) ** 2, dim=2).T
      # torch.min returns tuple of (values, indexes)
      min_dists = torch.min(dists, dim=1)[0]
      min_dists_x_weights += min_dists @ self.weights[r - 1].T
    out = self.biases + torch.sum(min_dists_x_weights, dim=0)
    return torch.sigmoid(out)

  def learn(self, x, labels, epochs=1000, lr=0.01):
    optimiser = optim.Adam([*self.shapelets, self.biases, *self.weights], lr=lr, weight_decay=self.lambda_w)
    loss = nn.BCELoss()
    self.pregenerate_segment_idxs(x.shape[1])
    epoch_losses = np.zeros((epochs))
    labels = torch.tensor(labels, dtype=torch.float64)
    for e in range(epochs):
      for i in range(x.shape[0]):
        optimiser.zero_grad()
        out = self.forward(x[i])
        total_loss = torch.sum(loss(out, labels[i]))
        epoch_losses[e] += total_loss
        total_loss.backward()
        optimiser.step()
      print(f"epoch: {e + 1} loss: {epoch_losses[e]}")

    print(f"Saving weights in ['{self.save_shapelets_loc}', '{self.save_weights_loc}', '{self.save_biases_loc}']")
    torch.save(self.shapelets, self.save_shapelets_loc)
    torch.save(self.weights, self.save_weights_loc)
    torch.save(self.biases, self.save_biases_loc)
