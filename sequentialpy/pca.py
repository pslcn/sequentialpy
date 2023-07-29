import numpy as np

# principal component analysis - eigenvectors of the covariance matrix
def pca(datapoints, components_k=100):
	# datapoints = datapoints.reshape((datapoints.shape[0], datapoints.shape[2] ** 2))
	datapoints -= np.mean(datapoints, axis=0)
	cov_mat = np.cov(datapoints, rowvar=False)
	eig_vecs = np.sort(np.linalg.eig(cov_mat)[1])[::-1]
	eig_vecs_k = eig_vecs[:, :components_k]
	return np.dot(eig_vecs_k.T, datapoints.T).T

X_with_idxs = lambda X: np.concatenate((X[:, np.newaxis, :], np.repeat(np.arange(0, X.shape[1]).reshape((1, 1, -1)), X.shape[0], axis=0)), axis=1)
