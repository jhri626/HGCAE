# Graph Isometry Regularizer

This note documents the GGAE-style graph isometry regularizer added to HGCAE.
The default weight is zero, so existing scripts keep their original behavior
unless `--lambda-iso` is set to a positive value.

## Files Changed

- `config.py`
  - Adds CLI flags for the regularizer:
    - `--lambda-iso`
    - `--iso-sample-size`
    - `--iso-bandwidth`
    - `--iso-bandwidth-method`
    - `--iso-measure`
    - `--iso-chart`
- `utils/isometry_utils.py`
  - Implements the graph-distance heat-kernel Laplacian.
  - Implements the `J G_data^{-1} J^T` estimator.
  - Implements Poincare and Euclidean chart metrics.
  - Implements the distortion losses.
- `utils/data_utils.py`
  - When `--lambda-iso > 0`, computes and caches shortest-path distances from
    `adj_train`.
  - The cache path is the same split cache directory used for edge masks:
    `/root/tmp/<task>/<dataset>/seed<seed>-val<val>-test<test>/graph_dist.npy`.
- `models/base_models.py`
  - Adds `lambda_iso * L_iso` to the training loss in `LPModel.compute_metrics`.
  - The regularizer is applied only for `split == "train"`.

## Objective

Before this change, HGCAE optimized:

```text
L_total = lambda_lp * L_lp + lambda_rec * L_rec
```

With the new regularizer:

```text
L_total = lambda_lp * L_lp + lambda_rec * L_rec + lambda_iso * L_iso
```

where:

- `L_lp` is the Fermi-Dirac link prediction BCE loss.
- `L_rec` is the feature reconstruction MSE.
- `L_iso` is the graph-distance local isometry distortion loss.

The implementation preserves the existing behavior when `lambda_rec == 0`: it
adds `lambda_iso * L_iso` to the loss value already used by HGCAE.

## Existing HGCAE Terms

The encoder maps graph data to latent embeddings:

```text
Z = f_theta(X, A_train)
z_i in B_c^d
```

For PoincareBall, HGCAE uses the ball:

```text
||z|| < 1 / sqrt(c)
```

The Poincare distance used by the Fermi-Dirac decoder is:

```text
d_c(z_i, z_j)
  = 2 / sqrt(c) * artanh(sqrt(c) * ||(-z_i) oplus_c z_j||)
```

The link probability is:

```text
p_ij = 1 / (exp((d_c(z_i, z_j)^2 - r) / t) + 1)
```

The link prediction loss is:

```text
L_lp = BCE(p_ij, 1) + BCE(p_ik, 0)
```

The reconstruction loss is:

```text
X_hat = g_phi(Z, A_train)
L_rec = MSE(X_hat, X)
```

## Isometry Regularizer

The regularizer uses graph shortest-path distance as the data geometry:

```text
D^G_ij = shortest_path_distance_G(i, j)
```

For link prediction, this is computed from `adj_train`, after validation and
test edges have been masked out. This avoids leaking held-out edges into the
regularizer. For node clustering, `adj_train` is the full graph because
`--node-cluster 1` sets `val_prop = test_prop = 0`.

For a sampled node set `S` of size `m`, the implementation slices:

```text
D_S in R^{m x m}
Z_S in R^{m x d}
```

If `--iso-sample-size 0`, all nodes are used.

### Heat-Kernel Graph Laplacian

With fixed bandwidth `h = --iso-bandwidth`:

```text
K_ij = exp(-(D^G_ij)^2 / h)
```

The normalized kernel is:

```text
q_i = sum_j K_ij
K_tilde_ij = K_ij / (q_i q_j)
q_tilde_i = sum_j K_tilde_ij
```

The graph Laplacian/generator used by GGAE is:

```text
L_ij = (K_tilde_ij / q_tilde_i - delta_ij) / ((1/4) h)
```

This corresponds to `compute_graph_laplacian` in
`utils/isometry_utils.py`.

### Local Metric Estimate

For latent points `z_i`, the code estimates:

```text
H_i = 1/2 * sum_j L_ij (z_j - z_i)(z_j - z_i)^T
```

This is equivalent to the `J G_data^{-1} J^T` estimate used by GGAE:

```text
H_i ~= J_i G_data^{-1} J_i^T
```

This corresponds to `get_jginvjt` in `utils/isometry_utils.py`.

### Poincare Chart Metric

For PoincareBall with curvature `-c`, the chart metric tensor is:

```text
lambda_c(z) = 2 / (1 - c ||z||^2)
G_P(z) = lambda_c(z)^2 I
```

This matches HGCAE's Poincare convention. The implementation uses the final
embedding curvature from the last hyperbolic layer:

```text
curvature = self.encoder.layers[-1].hyp_act.c_out
```

This corresponds to `poincare_metric_matrix` in `utils/isometry_utils.py` and
`get_embedding_curvature` in `models/base_models.py`.

### Distortion Measure

The chart-corrected local metric is:

```text
M_i = H_i G_P(z_i)
```

For an isometric embedding:

```text
M_i ~= I
```

The default distortion is:

```text
L_iso = mean_i tr(M_i^2) / mean_i(tr(M_i))^2
```

This is selected by:

```text
--iso-measure relaxed_distortion
```

Other supported measures are:

```text
least_squares_distortion:
  L_iso = mean_i [tr(M_i^2) - 2 tr(M_i)]

harmonic_mapping_distortion:
  L_iso = mean_i tr(M_i)
```

These correspond to `distortion_measure` in `utils/isometry_utils.py`.

## Training Usage

Example Cora link prediction run with the regularizer:

```bash
python train_solver.py \
  --model HGCAE \
  --dataset cora \
  --manifold PoincareBall \
  --cuda 0 \
  --lambda-iso 0.1 \
  --iso-sample-size 512 \
  --iso-bandwidth 4.0 \
  --iso-measure relaxed_distortion
```

The existing shell scripts can be modified by adding the same flags to the
`python train_solver.py` command.

## Practical Notes

- Start with small graphs such as Cora or Citeseer.
- `graph_dist.npy` is dense and can be large. The code loads it with
  `mmap_mode="r"` and moves only the sampled submatrix to GPU.
- For large datasets, keep `--iso-sample-size` small, for example `256` or
  `512`.
- Recommended first sweep:

```text
lambda_iso:      0.001, 0.01, 0.1
iso_bandwidth:   2.0, 4.0, 8.0
iso_sample_size: 256, 512
iso_measure:     relaxed_distortion
```

## Code-to-Formula Map

```text
D^G shortest paths
  -> utils/data_utils.py
  -> compute_shortest_path_distance(adj_train)

K_ij and L_ij
  -> utils/isometry_utils.py
  -> compute_graph_laplacian(...)

H_i = 1/2 sum_j L_ij (z_j - z_i)(z_j - z_i)^T
  -> utils/isometry_utils.py
  -> get_jginvjt(...)

G_P(z_i) = (2 / (1 - c ||z_i||^2))^2 I
  -> utils/isometry_utils.py
  -> poincare_metric_matrix(...)

L_iso
  -> utils/isometry_utils.py
  -> graph_isometry_loss(...)

L_total += lambda_iso * L_iso
  -> models/base_models.py
  -> LPModel.compute_metrics(...)
```
