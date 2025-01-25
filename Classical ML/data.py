# File to generate different datasets for ML algos to be tested

import sklearn.datasets as skd
import numpy as np


class DataGenerator:
    def __init__(self, n_samples=200, dataset_name="moons"):
        self.n_samples = n_samples
        self.dataset_name = dataset_name

    def moons(self):
        X, y = skd.make_moons(n_samples=self.n_samples, noise=0.2, random_state=42)
        return X, y

    def circles(self):
        X, y = skd.make_circles(n_samples=self.n_samples, random_state=42)
        return X, y

    def generate(self):
        if self.dataset_name.lower() == "moons":
            return self.moons()

        if self.dataset_name.lower() == "circles":
            return self.circles()
