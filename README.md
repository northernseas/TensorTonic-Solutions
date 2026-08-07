# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Theo Lepage's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/theolepage.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Aggregation Functions | Compute selected NumPy aggregation functions globally or along a requested axis using float64 values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-aggregation |
| Angle Features | Return a float64 array where row 0 contains the sine values, row 1 the cosine values, and row 2 the tangent values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-angle-features |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-basic-indexing |
| Boolean Masking | Build three filtered views of a 2D array: an element-level boolean mask, rows kept when any element exceeds a threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-boolean-masking |
| Column Scaling | Scale every column of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-col-scaling |
| Concat and Correlate | Concatenate two 2-D arrays row-wise and return a (3, n, n) stack of Pearson correlation matrices: one for each input and one for the combined data. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-concat-correlate |
| Create Arrays from Lists | Create NumPy arrays from Python lists with the requested dtype and return their values, shape, dimensions, and element count. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-create-array |
| Fancy Indexing | Convert the data to float64 and return the array formed by selecting elements along that axis using integer array indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-fancy-indexing |
| Filter and Extract | Implement Filter and Extract, and apply a boolean mask to select values strictly greater than threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-filter-extract |
| Mutation Trap | Extract an independent NumPy row copy, mutate it safely, and verify that the original array remains unchanged. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-mutation-trap |
| Normalized Difference | Use two 2D arrays a and b of the same shape and a scalar range [lo, hi], clip both arrays to [lo, hi], rescale each to [0, 1]. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-diff |
| Norm-Gated Linear Transform | Compute the linear transform Z = X @ W, then zero out every row of Z whose L2 norm is strictly below the threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-gate |
| Normalize Columns | Standardize each NumPy matrix column by subtracting its mean and dividing by its population standard deviation. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-normalize-columns |
| Outer Sum | Compute the broadcasted outer sum of two NumPy vectors without loops, supporting different lengths and numeric values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-outer-sum |
| Pairwise Differences | Implement Pairwise Differences, and compute the pairwise difference matrix without any Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-pairwise-diff |
| Quantize and Frame | Apply floor, ceiling, and nearest rounding to a NumPy matrix, then add a zero-valued border around each result. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-quantize-frame |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-random-arrays |
| Reshaping Arrays | Transform a float64 NumPy array with flattening, transposition, or a validated target shape. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-reshape |
| Row Extremes | Implement Row Extremes, using np.argmax(axis=1) to find the column index of the maximum value in each row. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-extremes |
| Row Scaling | Scale every row of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-scaling |
| Sort and Argsort | Return NumPy values sorted along a selected axis together with the indices that produce the same ordering. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-sort-argsort |
| Tile and Diff | Tile a 2-D array vertically and return the tiled result alongside its row-wise finite differences, packed as a (2, m·reps, n) float64 array. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-tile-diff |
| Winsorize | Winsorization clips extreme values in each column to percentile-based bounds, a standard technique for suppressing outliers in ML preprocessing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-winsorize |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-zeros-ones |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-activation-function-from-scratch |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-basic-autograd |
| Batch Normalization | Normalize each feature across the batch, then scale and shift using learnable parameters. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-batch-normalization |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-build-simple-nn-from-scratch |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-custom-linear-layer |
| Dropout from Scratch | Implement PyTorch inverted dropout from a supplied mask during training while returning inputs unchanged in evaluation mode. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-dropout-from-scratch |
| Loss Functions | Implement three common loss functions from scratch using PyTorch tensor operations: mean squared error, cross-entropy, and Huber loss. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-loss-functions |
| Mini Training Loop | Run one complete PyTorch training epoch over a DataLoader and return the sample-weighted mean loss. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-mini-training |
| Softmax from Scratch | Implement numerically stable batched softmax in PyTorch by shifting logits before exponentiation and normalization. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-softmax-from-scratch |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-reshape |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/theolepage)
<!-- tensortonic:end -->
