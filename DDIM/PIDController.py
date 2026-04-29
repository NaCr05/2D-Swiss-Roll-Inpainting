"""
DDIM/PIDController.py
=====================
Non-parametric PID controller with SNR Sigmoid multiplicative gating.

Augmented MDP action: a_t = SNR_lock(t) ⊙ u_t ∈ R^2
  u_t = Kp * e_t + Ki * I_t + Kd * D_t   (standard PID output)
  SNR_lock = σ(beta_sigmoid * (SNR_t - theta_sigmoid))  ∈ [0,1]

State evolution (managed externally by AugmentedMDP):
  I_t   = clip(gamma * I_{prev} + e_t, -M, M)
  bar_e_t = mu_ema * bar_e_{prev} + (1 - mu_ema) * e_t
  D_t   = (bar_e_t - bar_e_{prev}) / |Δᾱ_t|
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class PIDController:
    def __init__(
        self,
        Kp: float = 0.05,
        Ki: float = 0.1,
        Kd: float = 0.005,
        gamma: float = 0.9,
        M: float = 1.0,
        mu_ema: float = 0.9,
        beta_sigmoid: float = 5.0,
        theta_sigmoid: float = 1.0,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.gamma = gamma
        self.M = M
        self.mu_ema = mu_ema
        self.beta_sigmoid = beta_sigmoid
        self.theta_sigmoid = theta_sigmoid

        self._I_mem: Optional[torch.Tensor] = None   # Integral memory I_t
        self._bar_e_prev: Optional[torch.Tensor] = None   # EMA previous smoothed error bar_e_{prev}
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        self._I_mem = None
        self._bar_e_prev = None
        self._initialized = False

    def compute_action(
        self,
        e_t: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        alpha_bar_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PID-guided action a_t for one DDIM step.

        Args:
            e_t:          Error vector, shape [N_mask, 2] or [2]
            alpha_bar_t:  Cumulative alpha at timestep t, scalar or shape []
            alpha_bar_prev: Cumulative alpha at timestep prev_t, scalar or shape []

        Returns:
            a_t:        Final action = SNR_lock ⊙ u_t, same shape as e_t
            u_t:        Raw PID output, same shape as e_t
            SNR_lock:   Sigmoid gating scalar, scalar
            bar_e_t:    EMA smoothed error at this step, same shape as e_t
            D_t:        Derivative term, same shape as e_t
        """
        device = e_t.device
        e_t_flat = e_t.reshape(-1, 2)            # [N, 2]

        I_t = self._get_I_mem(device, e_t_flat.shape[0])
        bar_e_prev = self._get_bar_e_prev(device, e_t_flat.shape[0])

        # ── EMA low-pass filter on error ──────────────────────────────
        bar_e_t = self.mu_ema * bar_e_prev + (1.0 - self.mu_ema) * e_t_flat

        # ── I term: decay + saturation ────────────────────────────────
        I_t = torch.clamp(self.gamma * I_t + e_t_flat, -self.M, self.M)

        # ── D term: real-time-scale derivative via EMA error ─────────
        if alpha_bar_prev is None:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        delta_alpha = torch.abs(alpha_bar_t - alpha_bar_prev)
        # Guard: when t == prev_t (last step), delta_alpha = 0 → skip D term
        D_t = torch.where(
            delta_alpha > 1e-6,
            (bar_e_t - bar_e_prev) / delta_alpha.clamp(min=1e-8),
            torch.zeros_like(bar_e_t)
        )

        # ── Standard PID output ───────────────────────────────────────
        u_t = self.Kp * e_t_flat + self.Ki * I_t + self.Kd * D_t

        # ── SNR Sigmoid multiplicative gate ──────────────────────────
        snr_t = torch.sqrt(
            alpha_bar_t / (1.0 - alpha_bar_t.clamp(min=1e-6))
        )
        snr_lock = torch.sigmoid(self.beta_sigmoid * (snr_t - self.theta_sigmoid))

        a_t = snr_lock * u_t

        # ── Persist state for next step ───────────────────────────────
        self._I_mem = I_t.detach()
        self._bar_e_prev = bar_e_t.detach()
        self._initialized = True

        # Restore original shape
        orig_shape = e_t.shape
        a_t = a_t.reshape(orig_shape)
        u_t = u_t.reshape(orig_shape)
        bar_e_t = bar_e_t.reshape(orig_shape)
        D_t = D_t.reshape(orig_shape)

        return a_t, u_t, snr_lock, bar_e_t, D_t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_I_mem(self, device: torch.device, batch_size: int) -> torch.Tensor:
        if self._I_mem is not None:
            return self._I_mem.to(device)
        return torch.zeros(batch_size, 2, device=device)

    def _get_bar_e_prev(self, device: torch.device, batch_size: int) -> torch.Tensor:
        if self._bar_e_prev is not None:
            return self._bar_e_prev.to(device)
        return torch.zeros(batch_size, 2, device=device)

    def get_state_norms(self) -> dict[str, float]:
        """Return current state norms for logging."""
        return {
            "I_norm": float(self._I_mem.norm().item()) if self._I_mem is not None else 0.0,
            "bar_e_norm": float(self._bar_e_prev.norm().item()) if self._bar_e_prev is not None else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "gamma": self.gamma,
            "M": self.M,
            "mu_ema": self.mu_ema,
            "beta_sigmoid": self.beta_sigmoid,
            "theta_sigmoid": self.theta_sigmoid,
        }
