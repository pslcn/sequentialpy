import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sequentialpy import shapelets

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

	if 1:
		print(shapelet_transform.transformed_representation(test_X[0]))

	else:
		for i in range(test_X.shape[0]):
			pred = shapelet_transform.forward(test_X[i])
			print(f"[{i}] pred: {pred.data} actual: {test_labels[i]}")

def plot_vl_shapelets(shapelet_transform, length_scales, shapelet_min_length, num_shapelets):
	import math

	if length_scales >= 3:
		col_nelems = math.ceil(length_scales ** 0.5)
		fig, ax = plt.subplots(math.ceil(length_scales / col_nelems), col_nelems)
	else:
		fig, ax = plt.subplots(length_scales, 1)

	fig.suptitle(f"Number of Shapelets: {num_shapelets}")
	for r, ax_r in enumerate([an_ax for row in ax for an_ax in row] if ax.ndim == 2 else [an_ax for an_ax in ax]):
		if r < length_scales:
			ax_r.set_title(f"Shapelet Length: {(r + 1) * shapelet_min_length}")
			for shapelet in shapelet_transform.shapelets[r].detach().numpy():
				ax_r.plot(shapelet)
	else:
		ax_r.axis("off")

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	num_shapelets, shapelet_min_length, length_scales = 1, 25, 5
	shapelet_transform = shapelets.Shapelets(num_shapelets, 1, shapelet_min_length, length_scales=length_scales, load_weights=True)

	# train(shapelet_transform, epochs=1000, lr=1e-5)
	test(shapelet_transform)
	# plot_vl_shapelets(shapelet_transform, length_scales, shapelet_min_length, num_shapelets)
