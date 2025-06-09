#
# Performing PCA on tensors with all frequencies and indices
#

import os, argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch import Tensor

def parse_args():
    parser = argparse.ArgumentParser(
        prog="pca.py",
        description="Performs PCA on tensors.",
    )
    parser.add_argument("-d", "--dirs-in", nargs="+", required=True,
                        help="Tensor directory or directories")
    parser.add_argument("-n", "--n_components", type=int, default=None, help="Number of principal components.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    min_n = 2 if args.n_components is None else args.n_components
    max_n = 15 if args.n_components is None else args.n_components + 1
    
    data = None
    
    variances = np.zeros((max_n, max_n))
    cum_var = np.zeros((max_n, max_n))
    
    for n in range(min_n, max_n):
        pca = PCA(n_components=n)
        X_pca = pca.fit(data)
        variances[n] = pca.explained_variance_ratio_
        cum_var[n] = np.cumsum(pca.explained_variance_ratio_)
    variances = variances[n:, :]
    cum_var = cum_var[n:, :]
        
    n_figs = max_n - min_n
    fig, axes = plt.subplot(2, max_n-min_n, figsize=(15, 15))
    for i in range(max_n - min_n):
        n = i + min_n
        a1 = axes[0, i]
        a2 = axes[1, i]
        a1.plot(range(n, max_n), variances[n, :], color="blue")
        a1.set_title(f"Variances for {n} Components")
        a2.plot(range(n, max_n), cum_var[n, :], color="blue")
        a2.set_title(f"Cumulative Variance for {n} Components")
    plt.tight_layout()
    plt.show()    
    
if __name__ == "__main__":
    main()