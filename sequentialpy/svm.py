import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

  # def idx_closest_to_line(self, points, line):
  #   perpendicular_grad = -(1 / line[1])
  #   dists = []
  #   for point in points:
  #     perpendicular_c = self.calc_y_intercept(perpendicular_grad, point)
  #     intersect_x = (perpendicular_c - line[0]) / (line[1] - perpendicular_grad)
  #     intersect_y = line[0] + line[1] * intersect_x
  #     dists.append(torch.sqrt((point[1] - intersect_y) ** 2 + (point[0] - intersect_x) ** 2))
  #   return torch.argmin(dists)




class TorchLinearSVM:
  def __init__(self, clusters, centroids, n_iters=20, lr=0.01):
    centroids = torch.tensor(centroids)
    midpoint = torch.mean(centroids, dim=0)
    self.hyperplane_m = torch.tensor([-(1 / torch.divide(*torch.flip(torch.subtract(*centroids), dims=(0,))))], requires_grad=True, dtype=torch.float64)
    self.hyperplane_c = torch.tensor(self.calc_y_intercept(self.hyperplane_m, midpoint), requires_grad=True, dtype=torch.float64)
    self.support_vecs = torch.zeros((2, 2), requires_grad=False, dtype=torch.float64)

    optimiser = optim.Adam([self.hyperplane_m, self.hyperplane_c], lr=lr) 
    # loss = nn.MSELoss()
    for i in range(n_iters):
      optimiser.zero_grad()
      out = self.forward(clusters)
      # total_loss = loss(out)
      # total_loss.backward()
      # optimiser.step()

  def calc_y_intercept(self, grad, point): return point[1] - grad * point[0]

  def idx_closest(self, points):
    perpendicular_grad = -(1 / self.hyperplane_m)
    dists = []
    for point in points:
      perpendicular_c = self.calc_y_intercept(perpendicular_grad, point)
      intersect_x = (perpendicular_c - self.hyperplane_c) / (self.hyperplane_m - perpendicular_grad)
      intersect_y = self.hyperplane_c + self.hyperplane_m * intersect_x
      dists.append(torch.sqrt((point[1] - intersect_y) ** 2 + (point[0] - intersect_x) ** 2))
    return torch.argmin(torch.tensor(dists))

  def generate_support_vecs(self, clusters):
    self.support_vecs[:, 1] = self.hyperplane_m
    self.support_vecs[0, 0] = self.calc_y_intercept(self.support_vecs[0, 1], clusters[0][self.idx_closest(clusters[0])])
    self.support_vecs[1, 0] = self.calc_y_intercept(self.support_vecs[1, 1], clusters[1][self.idx_closest(clusters[1])])

  def calc_support_vector_margin(self):
    return torch.abs(self.support_vecs[0, 0] - self.support_vecs[1, 0]) / torch.sqrt(self.support_vecs[0, 1] ** 2 + self.support_vecs[1, 1] ** 2)

  def forward(self, clusters):
    self.generate_support_vecs(clusters)
    margin = self.calc_support_vector_margin()
    return margin
