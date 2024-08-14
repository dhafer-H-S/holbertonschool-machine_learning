t-SNE Implementation
This repository contains an implementation of the t-SNE (t-distributed Stochastic Neighbor Embedding) algorithm for dimensionality reduction. The code is structured into several functions that handle different parts of the t-SNE process, from initial dimensionality reduction with PCA, calculating affinities, to optimizing the low-dimensional embedding using gradient descent.

Table of Contents
Overview
Functions
1. PCA (pca)
2. Initialize t-SNE Variables (P_init)
3. Entropy and P Affinities (HP)
4. Symmetric P Affinities (P_affinities)
5. Q Affinities (Q_affinities)
6. Gradients (grads)
7. Cost Function (cost)
8. Full t-SNE Algorithm (tsne)
Example Usage
Conclusion
Overview
t-SNE is a technique for dimensionality reduction that is particularly well-suited for visualizing high-dimensional datasets. This implementation of t-SNE includes:

Dimensionality reduction using PCA.
Calculation of affinities (P and Q) between data points.
Optimization of a low-dimensional representation via gradient descent.
Use of techniques like early exaggeration and momentum to improve optimization.
Functions
1. Principal Component Analysis (PCA)
pca(X, idims)
Performs initial dimensionality reduction using PCA before applying t-SNE.

Input:
X: numpy.ndarray of shape (n, d) containing the dataset.
idims: Target dimensionality after PCA.
Output: Reduced dataset with shape (n, idims).
2. Initializing t-SNE Variables - P_init
P_init(X, perplexity)
Initializes variables required for t-SNE including pairwise distance matrix D, affinities P, betas β, and entropy H.

Input:
X: numpy.ndarray of shape (n, d) for the dataset.
perplexity: Perplexity value for the Gaussian distributions.
Output:
D: Pairwise distance matrix (n, n).
P: Zero-initialized affinities matrix.
betas: Array initialized to 1's.
H: Shannon entropy.
3. Entropy and P Affinities - HP
HP(Di, beta)
Calculates Shannon entropy and P affinities relative to a specific data point.

Input:

Di: Pairwise distances between one data point and all others (excluding itself).
beta: Beta value for the Gaussian distribution.
Output:

Hi: Shannon entropy for the point.
Pi: P affinities for the point.
4. Calculating Symmetric P Affinities - P_affinities
P_affinities(X, tol=1e-5, perplexity=30.0)
Performs a binary search to calculate symmetric P affinities for the dataset.

Input:
X: numpy.ndarray of shape (n, d) containing the dataset.
tol: Tolerance for Shannon entropy from perplexity.
perplexity: Desired perplexity value.
Output:
P: Symmetric affinities matrix (n, n).
5. Calculating Q Affinities - Q_affinities
Q_affinities(Y)
Calculates Q affinities in the low-dimensional space.

Input:
Y: numpy.ndarray of shape (n, ndim) containing the low-dimensional embedding.
Output:
Q: Q affinities matrix (n, n).
num: The numerator of the Q affinities formula.
6. Calculating Gradients - grads
grads(Y, P)
Computes the gradients of Y based on the P affinities and Q affinities.

Input:

Y: Low-dimensional representation of the data.
P: P affinities matrix (n, n).
Output:

dY: Gradients of Y.
Q: Q affinities matrix.
7. Cost Function - cost
cost(P, Q)
Calculates the cost of the t-SNE transformation using the KL-divergence between P and Q.

Input:
P: P affinities matrix (high-dimensional).
Q: Q affinities matrix (low-dimensional).
Output:
C: The cost of the transformation (KL-divergence).
8. t-SNE Transformation - tsne
tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500)
Performs the complete t-SNE transformation on the dataset.

Input:

X: Dataset (n, d) to be transformed.
ndims: Target dimensionality for the low-dimensional embedding.
idims: Dimensionality after PCA.
perplexity: Perplexity for the Gaussian distributions.
iterations: Number of iterations for optimization.
lr: Learning rate for gradient descent.
Output:

Y: Low-dimensional representation of the data.
Process:

PCA: Reduce dimensions using PCA.
P Affinities: Calculate P affinities in high-dimensional space.
Early Exaggeration: Exaggerate P affinities for the first 100 iterations.
Gradient Descent: Update low-dimensional embedding using gradients.
Re-centering: Re-center Y after each iteration.
Cost Logging: Print the cost every 100 iterations.
Example Usage
python
Copy code
import numpy as np

# Example dataset with 100 points in 50 dimensions
X = np.random.randn(100, 50)

# Run t-SNE on the dataset
Y = tsne(X, ndims=2, idims=30, perplexity=30.0, iterations=1000, lr=500)

# Y now contains the low-dimensional embedding
print(Y)
Conclusion
This t-SNE implementation effectively reduces high-dimensional data to a low-dimensional space, making it suitable for visualization or other dimensionality reduction tasks. The code incorporates key techniques such as PCA, P/Q affinities, early exaggeration, momentum, and gradient descent to provide an accurate, low-dimensional representation of high-dimensional data.

This file provides a clear overview of the t-SNE implementation and can be used as a guide for anyone looking to understand the individual components or run the algorithm on their own data.