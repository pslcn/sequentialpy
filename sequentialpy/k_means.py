import numpy as np
import numba as nb

@nb.njit 
def euclidean_distance(a, b):
  return np.sqrt(np.sum((a - b) ** 2))

@nb.njit
def k_means_clusters_step(centroids, clusters_k, datapoints, nelems):
  clusters = np.zeros((clusters_k, nelems, 2))
  euclidean_dists = np.zeros((nelems, clusters_k))
  for d in range(nelems):
    for k in range(clusters_k):
      euclidean_dists[d][k] = euclidean_distance(centroids[k], datapoints[d])
  closest_centroids = np.argmin(euclidean_dists, axis=1)

  nelem_counters = np.zeros((clusters_k), dtype=np.int64)
  for d in range(nelems):
    cluster_idx = closest_centroids[d]
    clusters[cluster_idx][nelem_counters[cluster_idx]] = datapoints[d]
    nelem_counters[cluster_idx] += 1
  clusters = [clusters[k][:nelem_counters[k]] for k in range(clusters_k)]
  return clusters

@nb.njit(parallel=True)
def k_means_with_centroids(clusters_k, datapoints, nelems, n_update_iters=6):
	shuffled_idxs = np.arange(0, nelems)
	np.random.shuffle(shuffled_idxs)
	centroids = datapoints[shuffled_idxs[:clusters_k]]
	for i in range(n_update_iters):
		clusters = k_means_clusters_step(centroids, clusters_k, datapoints, nelems)
		for k in range(clusters_k):
			centroids[k] = (np.mean(clusters[k][:, 0]), np.mean(clusters[k][:, 1]))
	return clusters, centroids
