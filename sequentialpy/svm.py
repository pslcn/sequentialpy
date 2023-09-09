import torch
import torch.optim as optim
import torch.nn as nn

class TorchLinearSVM:
  def __init__(self):
    self.hyperplane = None
    self.support_vecs = torch.zeros((2, 2), requires_grad=False, dtype=torch.float64)

  @property
  def hyperplane(self): return self.hyperplane.detach().numpy()

  # @property
  # def support_vecs(self): return self.support_vecs.detach().numpy()

  def calc_y_intercept(self, grad, point): return point[1] - grad * point[0]

  def idx_closest_to_line(self, points, line):
    perpendicular_grad = -(1 / line[1])
    dists = []
    for point in points:
      perpendicular_c = self.calc_y_intercept(perpendicular_grad, point)
      intersect_x = (perpendicular_c - line[0]) / (line[1] - perpendicular_grad)
      intersect_y = line[0] + line[1] * intersect_x
      dists.append(torch.sqrt((point[1] - intersect_y) ** 2 + (point[0] - intersect_x) ** 2))
    return torch.argmin(dists)

  def generate_support_vecs(self, clusters):
    self.support_vecs[:, 1] = self.hyperplane[1]
    self.support_vecs[0, 0] = self.calc_y_intercept(self.support_vecs[0, 1], clusters[0][self.idx_closest_to_line(clusters[0], self.hyperplane)])
    self.support_vecs[1, 0] = self.calc_y_intercept(self.support_vecs[1, 1], clusters[1][self.idx_closest_to_line(clusters[1], self.hyperplane)])

  def calc_support_vector_margin(self):
    return torch.abs(self.support_vecs[0, 0] - self.support_vecs[1, 0]) / torch.sqrt(support_vecs[0, 1] ** 2 + support_vecs[1, 1] ** 2)

  def forward(self, clusters):
    self.generate_support_vecs(clusters)
    margin = self.calc_support_vector_margin()
    return margin

  def learn_with_k_means(self, clusters, centroids, niters=100, lr=0.01):
    if self.hyperplane is None:
      self.hyperplane = torch.zeros((2), requires_grad=True, dtype=torch.float64)

      self.hyperplane[1] = -(1 / torch.divide(*torch.subtract(*centroids)[::-1]))
      midpoint = torch.mean(centroids, axis=0)
      self.hyperplane[0] = self.calc_y_intercept(self.hyperplane[1], midpoint)

    optimiser = optim.Adam([self.hyperplane], lr=lr) 
    loss = nn.MSELoss()
    for i in range(niters):
      optimiser.zero_grad()
      out = self.forward(clusters)
      total_loss = loss(out)
      total_loss.backward()
      optimiser.step()
