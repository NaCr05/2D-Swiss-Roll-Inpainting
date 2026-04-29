"""
DDIM/AugmentedMDP.py
====================
Augmented MDP state maintenance, reward computation, and transition wrapper.

State space S_t = [x_t; I_t; D_t; e_t; bar_e_t] ∈ R^10
  x_t:     Current noisy latent, dim 2
  I_t:     Integral memory (clipped), dim 2
  D_t:     Derivative trend (EMA-scale), dim 2
  e_t:     Instantaneous error (autograd gradient), dim 2
  bar_e_t: EMA-smoothed error, dim 2

Transition (PINN-guided masked DDIM):
  xhat_0  = Tweedie(x_t, eps_θ)                   ← manifold estimate
  x_{t-1} = DDIM_step(xhat_0, t→t-1)             ← one DDIM reverse step
  x_{t-1}* = (1-λ)·x_{t-1} + λ·nearest_known    ← direct manifold pull (full strength)

Reward:
  r_t = -loss_boundary = -MSE(边界x̂₀ ↔ 最近邻已知点)
"""

import torch
from typing import Tuple, Optional, List

from DDIM.PIDController import PIDController
from DDIM.BoundaryMetrics import compute_boundary_loss


class AugmentedMDP:
    def __init__(
        self,
        pid_controller: PIDController,
        x_known: torch.Tensor,
        boundary_indices: Optional[List[int]] = None,
        x_GT_masked: Optional[torch.Tensor] = None,
    ):
        self.pid = pid_controller
        self.x_known = x_known
        self.boundary_indices = boundary_indices or []
        self.x_GT_masked = x_GT_masked

        self._x_cur: Optional[torch.Tensor] = None
        self._step_count: int = 0

    def reset(self, x_init: torch.Tensor) -> torch.Tensor:
        self.pid.reset()
        self._x_cur = x_init.detach().clone()
        self._step_count = 0
        return self._x_cur

    def step(
        self,
        x_mask: torch.Tensor,
        model: torch.nn.Module,
        alpha_bar: torch.Tensor,
        t: int,
        prev_t: int,
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        One step of the Augmented MDP for masked-region inpainting.

        Core loop:
          1. Noise prediction for masked region only
          2. Tweedie estimate xhat_0 = (x_t - √(1-ᾱ)·ε_θ) / √ᾱ
          3. DDIM one-step: x_{t-1} from xhat_0
          4. Direct manifold pull: x_{t-1}* = (1-λ)·x_{t-1} + λ·nearest_known
             λ is large (0.7) to bring the result toward the Swiss Roll manifold

        Key insight: we correct x_{t-1} directly, NOT xhat_0.
        This avoids the √(1-ᾱ) decay that prevented corrections from propagating
        in previous versions.

        Args:
            x_mask:    Current masked latent, shape [N_mask, 2]
            model:     Frozen DDIM noise predictor
            alpha_bar: Cumulative alpha bar, shape [T]
            t:         Current DDIM timestep
            prev_t:    Previous DDIM timestep

        Returns:
            x_next:  Next masked latent x_{t-1} (after manifold pull)
            r_t:     Scalar reward for this step
            info:    Dict with all state metrics
        """
        device = x_mask.device
        N_mask = x_mask.shape[0]

        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_1m_ab_t = torch.sqrt((1.0 - ab_t).clamp(min=1e-8))
        sqrt_ab_prev = torch.sqrt(ab_prev)
        sqrt_1m_ab_prev = torch.sqrt((1.0 - ab_prev).clamp(min=1e-8))

        # ── Step 1: Noise prediction for masked region ───────────────────
        with torch.no_grad():
            t_tensor = torch.full((N_mask,), t, device=device, dtype=torch.long)
            eps_pred = model(x_mask, t_tensor)

        # ── Step 2: Tweedie estimate of x̂₀ ─────────────────────────────
        xhat_0 = (x_mask - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t.clamp(min=1e-8)

        # ── Step 3: DDIM one-step from xhat_0 ──────────────────────
        with torch.no_grad():
            pred_dir = (x_mask - sqrt_ab_t * xhat_0) / sqrt_1m_ab_t
            x_ddim = sqrt_ab_prev * xhat_0 + sqrt_1m_ab_prev * pred_dir

        # ── Step 4: Direct manifold pull on x_{t-1} ───────────────────
        # λ = clamp(1-ᾱ, 0.3, 0.9): large enough to pull toward manifold.
        # At t=196: λ≈0.7 → 70% toward nearest known.
        # Near t=0: λ≈0.7 → 70% toward nearest known (ᾱ≈0.98 always).
        # NOTE: λ is constant ≈ 0.7 because ᾱ ≈ 0.98 throughout.
        blend_lambda = float(torch.clamp(1.0 - ab_t, 0.3, 0.9).item())

        with torch.no_grad():
            nn_idx = torch.cdist(x_ddim, self.x_known).argmin(dim=1)
            x_nearest = self.x_known[nn_idx]

        x_next = (1.0 - blend_lambda) * x_ddim + blend_lambda * x_nearest

        # ── Step 5: Boundary loss (computed on xhat_0 for reward) ───────
        loss_boundary = self._compute_boundary_loss(xhat_0)
        r_t = float(-loss_boundary.item()) if loss_boundary is not None else 0.0

        # ── Step 6: PID state update (history tracking only) ───────────
        grad_per_point = self._get_boundary_grad_per_point(xhat_0)
        e_t_scalar = grad_per_point.mean(dim=0, keepdim=True)
        ab_prev_for_pid = ab_prev if prev_t >= 0 else torch.tensor(1.0, device=device)
        a_t_dummy, u_t_dummy, snr_lock, bar_e_t, D_t = self.pid.compute_action(
            e_t_scalar, ab_t, ab_prev_for_pid
        )

        # ── Step 7: Build info dict ─────────────────────────────────────
        info = {
            "t": t,
            "prev_t": prev_t,
            "r_t": r_t,
            "e_t_norm": float(grad_per_point.norm().item()),
            "u_t_norm": float(u_t_dummy.norm().item()),
            "a_t_norm": float(a_t_dummy.norm().item()),
            "snr_lock": float(snr_lock.item()),
            "bar_e_norm": float(bar_e_t.norm().item()),
            "D_t_norm": float(D_t.norm().item()),
            "blend_lambda": blend_lambda,
            "mse_t": float(
                ((x_next - self.x_GT_masked) ** 2).mean().item()
                if self.x_GT_masked is not None
                else 0.0
            ),
        }

        self._x_cur = x_next.detach()
        self._step_count += 1
        return x_next, r_t, info

    def _compute_boundary_loss(self, xhat_0: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.boundary_indices or len(self.boundary_indices) == 0:
            return None
        return compute_boundary_loss(xhat_0, self.x_known, self.boundary_indices)

    def _get_boundary_grad_per_point(
        self,
        xhat_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ∂L_boundary/∂xhat_0[i] for ALL mask points.

        L_boundary = mean over boundary points of ||x_boundary - nearest_known||²
        Soft Gaussian nearest-neighbor: each boundary point connects to a
        weighted-average of known points (Gaussian kernel), giving smoother gradients.
        Then broadcast to all mask points via Gaussian proximity kernel.

        Shape: [N_mask, 2]
        """
        if not self.boundary_indices or len(self.boundary_indices) == 0:
            return torch.zeros_like(xhat_0)

        x_boundary = xhat_0[self.boundary_indices]
        N_boundary = x_boundary.shape[0]

        sigma = 0.5
        dist_sq = torch.cdist(x_boundary, self.x_known) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        x_nearest = weights @ self.x_known
        residual = x_boundary - x_nearest
        grad_boundary = 2.0 * residual / max(N_boundary, 1)

        dist_to_boundary = torch.cdist(xhat_0, x_boundary) ** 2
        boundary_weights = torch.exp(-dist_to_boundary / (2 * sigma ** 2))
        boundary_weights = boundary_weights / (boundary_weights.sum(dim=1, keepdim=True) + 1e-8)
        grad_per_point = boundary_weights @ grad_boundary
        return grad_per_point

    def get_state_norms(self) -> dict:
        return self.pid.get_state_norms()
