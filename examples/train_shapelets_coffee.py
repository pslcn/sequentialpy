import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sequentialpy import shapelets, k_means

COFFEE_TRAIN_LOC = "~/Downloads/UCR_TS_Archive_2015/Coffee/Coffee_TRAIN"
COFFEE_TEST_LOC = "~/Downloads/UCR_TS_Archive_2015/Coffee/Coffee_TEST"

def shuffle_dataset(train_X, train_labels):
  shuffled_idxs = np.arange(0, train_X.shape[0], dtype=int)
  np.random.shuffle(shuffled_idxs)
  return train_X[shuffled_idxs], train_labels[shuffled_idxs]

df = pd.read_csv(COFFEE_TRAIN_LOC)
train_X = df.iloc[:, 1:].to_numpy()
train_labels = df.iloc[:, 0].to_numpy().reshape((-1, 1))
train_X, train_labels = shuffle_dataset(train_X, train_labels)


num_shapelets, shapelet_min_length, length_scales = 2, 20, 4
shapelet_transform = shapelets.Shapelets(num_shapelets, 1, shapelet_min_length, length_scales=length_scales, load_weights=True)

shapelet_transform.learn(train_X, train_labels, epochs=2000, lr=1e-4)
