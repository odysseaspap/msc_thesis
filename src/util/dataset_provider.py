import numpy as np
from sklearn.model_selection import train_test_split as skl_train_test_split
from sklearn.utils import shuffle

from .data_wrangling import *


class DatasetProvider():
    """
    Prepares the given dataset for the training task.
    """

    def __init__(self, data, labels):
        self.data_train, self.data_test, self.labels_train, self.labels_test = self._train_test_split(data, labels)
        self._train_batch_index = 0
        self._test_batch_index = 0

    def train_batch_iterator(self, num_samples):
        return self._make_iterator(num_samples, self.data_train, self.labels_train)

    def test_batch_iterator(self, num_samples):
        return self._make_iterator(num_samples, self.data_test, self.labels_test)

    def _make_iterator(self, num_samples, data, labels):
        current_idx = 0
        while True:
            # Reset current index in case end was reached.
            if current_idx >= len(self.data_train):
                current_idx = 0
                data, labels = shuffle(data, labels)
            batch_begin = current_idx
            batch_end = current_idx + num_samples
            current_idx = batch_end + 1
            yield [data[batch_begin:batch_end], labels[batch_begin:batch_end]]

    def _split_data_and_labels(self, dataset):
        return dataset[:,:-1], dataset[:,-1:]

    def _train_test_split(self, data, labels):
        return skl_train_test_split(data, labels, shuffle=True, random_state=5, test_size=0.2)