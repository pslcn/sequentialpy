import numpy as np

def savitzky_golay(y, m, polynomial_order):
  z = np.arange((1 - m) // 2, (m - 2) / 2 + 1, dtype=np.int32)
  J = z.reshape((-1, 1)) ** np.arange(0, polynomial_order + 1)
  C = np.sum((np.linalg.inv(J.T @ J) @ J.T).T, axis=1, dtype=np.float16)
  j_idxs = np.arange(m // 2, len(y) - (m // 2), dtype=np.int32).reshape((-1, 1)) + z
  ret = np.sum(C * y[j_idxs], axis=1)
  # pad using nearest values
  padded = np.concatenate((np.repeat(y[0], m // 2), ret, np.repeat(y[-1], m // 2)))
  return padded
