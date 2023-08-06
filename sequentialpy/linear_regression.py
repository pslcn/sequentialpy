import numpy as np

def ar_coeffs(x, lag_k=1):
  design_mat = np.ones((lag_k + 1, x.shape[0] - lag_k))
  idx_array = np.arange(0, x.shape[0] - lag_k) + np.arange(0, lag_k).reshape((-1, 1))
  design_mat[1:] = x[idx_array]
  return np.linalg.inv(design_mat @ design_mat.T) @ (design_mat @ x[lag_k:])
