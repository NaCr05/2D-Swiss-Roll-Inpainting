"""
DDIM/BoundaryMetrics.py
=======================
Boundary error computation, autograd gradient extraction,
and all evaluation metrics for inpainting quality assessment.

Two categories:
  1. Per-step metrics (logged every DDIM step, for convergence curves)
  2. Final metrics (computed once after inpainting completes)
"""

import torch
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Mask region definition
# ---------------------------------------------------------------------------
# Rectangular mask: x ∈ (-0.5, 0.5), y ∈ (-2.0, 2.0)
MASK_X_MIN = -0.5
MASK_X_MAX =  0.5
MASK_Y_MIN = -2.0
MASK_Y_MAX =  2.0
EPS_BOUNDARY = 0.05   # boundary layer thickness ≈ 5% of mask width/height


def in_rect_mask(x: torch.Tensor) -> torch.Tensor:
    """Return boolean mask for points inside the rectangular inpainting region."""
    return (
        (x[:, 0] > MASK_X_MIN) & (x[:, 0] < MASK_X_MAX) &
        (x[:, 1] > MASK_Y_MIN) & (x[:, 1] < MASK_Y_MAX)
    )


def find_boundary_indices(x: torch.Tensor) -> List[int]:
    """
    Find indices of points that are inside the mask AND within
    EPS_BOUNDARY distance from any edge of the rectangle.
    """
    mask = in_rect_mask(x)
    idx = torch.where(mask)[0]

    boundary_list = []
    for i in idx:
        pt = x[i]
        d = min(
            pt[0].item() - MASK_X_MIN,
            MASK_X_MAX - pt[0].item(),
            pt[1].item() - MASK_Y_MIN,
            MASK_Y_MAX - pt[1].item(),
        )
        if d <= EPS_BOUNDARY:
            boundary_list.append(i.item())

    return boundary_list


# ---------------------------------------------------------------------------
# Boundary loss + autograd gradient
# ---------------------------------------------------------------------------

def compute_boundary_loss(
    xhat_0: torch.Tensor,
    x_known: torch.Tensor,
    boundary_indices: List[int],
) -> torch.Tensor:
    """
    Boundary-consistency loss for autograd gradient extraction.

    Args:
        xhat_0:         Tweedie prediction, shape [N_mask, 2]
        x_known:        Known (non-masked) points, shape [N_known, 2]
        boundary_indices: Indices of boundary-layer points in xhat_0

    Returns:
        loss_boundary:  Scalar loss (differentiable, attached to graph)
                        Can be passed to torch.autograd.grad() to get e_t
    """
    x_boundary = xhat_0[boundary_indices]

    with torch.no_grad():
        dist_matrix = torch.cdist(x_boundary, x_known)          # [N_b, N_k]
        nearest_idx = dist_matrix.argmin(dim=1)
        x_nearest = x_known[nearest_idx]                        # [N_b, 2]

    loss_boundary = ((x_boundary - x_nearest) ** 2).sum(dim=1).mean()
    return loss_boundary


def compute_e_t(
    xhat_0: torch.Tensor,
    x_known: torch.Tensor,
    boundary_indices: List[int],
) -> torch.Tensor:
    """
    Compute e_t = ∂loss_boundary/∂xhat_0 via autograd.
    Returns mean gradient over boundary points, shape [2].
    """
    xhat_grad = xhat_0.detach().requires_grad_(True)
    loss_b = compute_boundary_loss(xhat_grad, x_known, boundary_indices)

    grad, = torch.autograd.grad(
        outputs=loss_b,
        inputs=xhat_grad,
        retain_graph=False,
    )
    # grad shape: [N_boundary, 2]; mean over boundary points → [2]
    e_t = grad.mean(dim=0)
    return e_t


# ---------------------------------------------------------------------------
# Evaluation metrics (final, computed once after inpainting)
# ---------------------------------------------------------------------------

def compute_mse(x_inpaint: torch.Tensor, x_GT: torch.Tensor) -> float:
    return float(((x_inpaint - x_GT) ** 2).mean().item())


def compute_rmse(x_inpaint: torch.Tensor, x_GT: torch.Tensor) -> float:
    return float(((x_inpaint - x_GT) ** 2).mean().sqrt().item())


def _kde_density_ratio(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    bandwidth: float = 0.1,
    grid_n: int = 50,
) -> float:
    """
    Very simple KDE-based density ratio approximation.
    Samples a grid and compares densities at inpainted vs non-masked locations.
    """
    all_pts = np.concatenate([pts_a, pts_b], axis=0)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    x_grid = np.linspace(x_min, x_max, grid_n)
    y_grid = np.linspace(y_min, y_max, grid_n)
    xv, yv = np.meshgrid(x_grid, y_grid)
    grid_pts = np.stack([xv.ravel(), yv.ravel()], axis=1)

    def _kde_density(pts: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
        diff = grid[:, np.newaxis, :] - pts[np.newaxis, :, :]    # [G, N, 2]
        d2 = (diff ** 2).sum(axis=2)                              # [G, N]
        densities = np.exp(-d2 / (2 * h ** 2)).mean(axis=1)     # [G]
        return densities

    dens_a = _kde_density(pts_a, grid_pts, bandwidth)
    dens_b = _kde_density(pts_b, grid_pts, bandwidth)
    dens_a = dens_a / (dens_a.sum() + 1e-8)
    dens_b = dens_b / (dens_b.sum() + 1e-8)

    kl_ab = float(np.sum(dens_a * np.log((dens_a + 1e-8) / (dens_b + 1e-8))))
    kl_ba = float(np.sum(dens_b * np.log((dens_b + 1e-8) / (dens_a + 1e-8))))
    return 0.5 * (kl_ab + kl_ba)


def compute_kl_divergence(x_inpaint: np.ndarray, x_nonmask: np.ndarray) -> float:
    return _kde_density_ratio(x_inpaint, x_nonmask)


def compute_js_divergence(x_inpaint: np.ndarray, x_nonmask: np.ndarray) -> float:
    """Symmetric JS divergence via KDE ratio."""
    return _kde_density_ratio(x_inpaint, x_nonmask)


def compute_manifold_fidelity(
    x_inpaint: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Swiss Roll manifold constraint: for each point, its y-coordinate should
    be consistent with the y-coordinates of its nearest neighbors.
    Return mean |y_i - mean(y_neighbors)|.
    """
    dist = torch.cdist(x_inpaint, x_inpaint)
    _, nn_idx = dist.topk(k + 1, largest=False)
    nn_idx = nn_idx[:, 1:]

    y = x_inpaint[:, 1:2]                          # [N, 1]
    y_neighbors = x_inpaint[:, 1:2][nn_idx]        # [N, k]
    mean_y_neighbors = y_neighbors.mean(dim=1, keepdim=True)

    return float(torch.abs(y - mean_y_neighbors).mean().item())


def compute_boundary_smoothness(
    x_inpaint: torch.Tensor,
    x_known: torch.Tensor,
    boundary_indices: List[int],
) -> float:
    """
    Mean L2 distance from each boundary inpainted point to its nearest known point.
    """
    if len(boundary_indices) == 0:
        return 0.0

    x_boundary = x_inpaint[boundary_indices]
    dist = torch.cdist(x_boundary, x_known)
    nearest_dist = dist.min(dim=1)[0]
    return float(nearest_dist.mean().item())


def compute_mmd(
    x_inpaint: torch.Tensor,
    x_nonmask: torch.Tensor,
    sigma: float = 1.0,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    """
    def _rbf_kernel(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)     # [Na, Nb, 2]
        d2 = (diff ** 2).sum(dim=2)                # [Na, Nb]
        return torch.exp(-d2 / (2 * s ** 2))

    n_a = x_inpaint.shape[0]
    n_b = x_nonmask.shape[0]

    K_aa = _rbf_kernel(x_inpaint, x_inpaint, sigma).fill_diagonal_(0).sum() / (n_a * (n_a - 1))
    K_bb = _rbf_kernel(x_nonmask, x_nonmask, sigma).fill_diagonal_(0).sum() / (n_b * (n_b - 1))
    K_ab = _rbf_kernel(x_inpaint, x_nonmask, sigma).sum() / (n_a * n_b)

    return float((K_aa + K_bb - 2 * K_ab).item())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_all_final_metrics(
    x_inpaint: torch.Tensor,
    x_GT: torch.Tensor,
    x_known: torch.Tensor,
    x_nonmask: torch.Tensor,
    boundary_indices: List[int],
) -> dict:
    metrics = {}
    metrics["MSE"]              = compute_mse(x_inpaint, x_GT)
    metrics["RMSE"]             = compute_rmse(x_inpaint, x_GT)
    metrics["KL_divergence"]    = compute_kl_divergence(
        x_inpaint.detach().cpu().numpy(),
        x_nonmask.detach().cpu().numpy(),
    )
    metrics["JS_divergence"]    = compute_js_divergence(
        x_inpaint.detach().cpu().numpy(),
        x_nonmask.detach().cpu().numpy(),
    )
    metrics["manifold_fidelity"] = compute_manifold_fidelity(x_inpaint)
    metrics["boundary_smoothness"] = compute_boundary_smoothness(
        x_inpaint, x_known, boundary_indices
    )
    metrics["MMD"]              = compute_mmd(x_inpaint, x_nonmask)
    return metrics
