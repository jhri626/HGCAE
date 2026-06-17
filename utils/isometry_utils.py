"""Graph-distance isometry regularization utilities."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch


def compute_shortest_path_distance(adj):
    """Return dense unweighted shortest-path distances for a sparse graph."""
    if not sp.isspmatrix(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return shortest_path(adj, directed=False, unweighted=True).astype(np.float32)


def _finite_distances(dist):
    finite = (dist == dist) & (dist != float("inf")) & (dist != -float("inf"))
    if bool(finite.all().item()):
        return dist

    values = dist.masked_select(finite)
    if values.numel() > 0:
        replacement = values.max().clamp_min(1.0) * 2.0
    else:
        replacement = dist.new_tensor(1.0)
    return torch.where(finite, dist, replacement)


def compute_graph_laplacian(dist_xx, bandwidth_method="fixed", bandwidth=50.0, epsilon=1e-8):
    """Compute the normalized graph Laplacian used by the GGAE distortion loss."""
    dist_xx = _finite_distances(dist_xx)
    batch_size, n_nodes, _ = dist_xx.shape
    laplacian_scale = 1.0 / 4.0
    dist_sq = dist_xx.pow(2)

    if bandwidth_method == "fixed":
        kernel = torch.exp(-dist_sq / bandwidth)
    elif bandwidth_method == "median_heuristic":
        median = dist_xx.median(dim=-1)[0].clamp_min(epsilon)
        inv_sigma = torch.diag_embed(1.0 / median)
        kernel = torch.exp(-(inv_sigma @ dist_sq @ inv_sigma))
    else:
        raise ValueError("Unknown bandwidth_method: {}".format(bandwidth_method))

    degree = kernel.sum(dim=1).clamp_min(epsilon)
    degree_inv = torch.diag_embed(1.0 / degree)
    kernel_tilde = degree_inv @ kernel @ degree_inv
    degree_tilde_inv = torch.diag_embed(1.0 / kernel_tilde.sum(dim=1).clamp_min(epsilon))
    eye = torch.diag_embed(torch.ones(batch_size, n_nodes, dtype=dist_xx.dtype, device=dist_xx.device))
    generator = degree_tilde_inv @ kernel_tilde - eye

    if bandwidth_method == "fixed":
        return generator / (laplacian_scale * bandwidth)
    if bandwidth_method == "median_heuristic":
        return inv_sigma @ (generator / laplacian_scale) @ inv_sigma
    raise ValueError("Unknown bandwidth_method: {}".format(bandwidth_method))


def get_jginvjt(laplacian, latent):
    """Estimate J G_data^{-1} J^T at each sampled node."""
    batch_size = laplacian.shape[0]
    n_nodes = laplacian.shape[1]
    latent_dim = latent.shape[-1]

    y_col = latent.unsqueeze(-1).repeat(1, 1, 1, latent_dim)
    y_row = latent.unsqueeze(-2).repeat(1, 1, latent_dim, 1)

    term1 = y_col * y_row
    term1 = (laplacian @ term1.contiguous().view(batch_size, n_nodes, latent_dim * latent_dim))
    term1 = term1.view(batch_size, n_nodes, latent_dim, latent_dim)

    ly = laplacian @ latent
    term2 = y_col * ly.unsqueeze(-2).repeat(1, 1, latent_dim, 1)
    term3 = y_row * ly.unsqueeze(-1).repeat(1, 1, 1, latent_dim)
    return 0.5 * (term1 - term2 - term3)


def _identity_like_points(points):
    latent_dim = points.shape[-1]
    eye = torch.eye(latent_dim, dtype=points.dtype, device=points.device)
    return eye.view(1, 1, latent_dim, latent_dim).expand(
            points.shape[0], points.shape[1], latent_dim, latent_dim
    )


def poincare_metric_matrix(points, curvature, epsilon=1e-8):
    """Poincare ball metric tensor in HGCAE's curvature convention."""
    if not torch.is_tensor(curvature):
        curvature = points.new_tensor([curvature])
    curvature = curvature.to(dtype=points.dtype, device=points.device).clamp_min(epsilon)
    sq_norm = points.pow(2).sum(dim=-1, keepdim=True)
    conformal_factor = 2.0 / (1.0 - curvature * sq_norm).clamp_min(epsilon)
    return conformal_factor.pow(2).unsqueeze(-1) * _identity_like_points(points)


def euclidean_metric_matrix(points):
    return _identity_like_points(points)


def distortion_measure(metric, measure="relaxed_distortion", epsilon=1e-8):
    trace = metric.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    trace_sq = (metric @ metric).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    if measure == "least_squares_distortion":
        return trace_sq.mean() - 2.0 * trace.mean()
    if measure == "harmonic_mapping_distortion":
        return trace.mean()
    if measure == "relaxed_distortion":
        return trace_sq.mean() / trace.mean().clamp_min(epsilon).pow(2)
    raise ValueError("Unknown isometry distortion measure: {}".format(measure))


def graph_isometry_loss(
        embeddings,
        graph_dist,
        curvature,
        sample_size=512,
        bandwidth_method="fixed",
        bandwidth=4.0,
        measure="relaxed_distortion",
        chart="poincare",
        epsilon=1e-8):
    """Compute a sampled graph-distance isometry loss for latent embeddings."""
    n_nodes = embeddings.shape[0]
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= n_nodes:
        idx_np = np.arange(n_nodes)
    else:
        idx_np = np.random.choice(n_nodes, int(sample_size), replace=False)

    idx = torch.LongTensor(idx_np).to(embeddings.device)
    latent = embeddings[idx].unsqueeze(0)
    dist_np = np.asarray(graph_dist[np.ix_(idx_np, idx_np)], dtype=np.float32)
    dist = torch.from_numpy(dist_np).to(device=embeddings.device, dtype=embeddings.dtype).unsqueeze(0)

    laplacian = compute_graph_laplacian(
            dist,
            bandwidth_method=bandwidth_method,
            bandwidth=bandwidth,
            epsilon=epsilon,
    )
    h_tilde = get_jginvjt(laplacian, latent)

    if chart == "poincare":
        chart_metric = poincare_metric_matrix(latent, curvature, epsilon=epsilon)
    elif chart == "euclidean":
        chart_metric = euclidean_metric_matrix(latent)
    else:
        raise ValueError("Unknown isometry chart: {}".format(chart))

    return distortion_measure(h_tilde @ chart_metric, measure=measure, epsilon=epsilon)
