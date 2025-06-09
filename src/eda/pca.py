#
# Performing PCA on tensors with all frequencies and indices
#

import os, argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch import Tensor

def parse_args():
    pass

def main():
    for n in range(2, 15):
        pca = PCA(n_components=n)
    pass

if __name__ == "__main__":
    main()