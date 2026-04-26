import torch


class ReverseDiffusion:
    @staticmethod
    @torch.no_grad()
    def p_sample(model, x_t, t, betas, eta=0.0, clip_range=(-1.0, 1.0), clip_denoised=False):
        prev_t = t - 1

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=x_t.device)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))

        # --- DDIM: compute sigma_t ---
        # sigma_t = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_prev)
        if prev_t >= 0 and eta > 0:
            sigma_t = eta * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8))
                * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
            )
        else:
            sigma_t = torch.tensor(0.0, device=x_t.device)

        # --- 1. Predict noise with the model ---
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)

        # --- 2. Estimate x0 from x_t and predicted noise ---
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_theta) / sqrt_alpha_bar_t.clamp(min=1e-8)

        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, clip_range[0], clip_range[1])

        # --- 3. DDIM sampling formula ---
        # pred_dir = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
        pred_dir = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

        if prev_t >= 0:
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt((1.0 - alpha_bar_prev).clamp(min=0))

            x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * pred_dir

            # Add noise only when eta > 0
            if eta > 0:
                z = torch.randn_like(x_t)
                x_prev = x_prev + sigma_t * z
        else:
            x_prev = x_t

        return x_prev

    @staticmethod
    @torch.no_grad()
    def ddim_sample(model, x_t, t, prev_t, betas, eta=0.0, clip_range=(-1.0, 1.0), clip_denoised=False):
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=x_t.device)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))

        # --- DDIM: compute sigma_t ---
        if prev_t >= 0 and eta > 0:
            sigma_t = eta * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8))
                * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
            )
        else:
            sigma_t = torch.tensor(0.0, device=x_t.device)

        # --- 1. Predict noise ---
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)

        # --- 2. Estimate x0 ---
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_theta) / sqrt_alpha_bar_t.clamp(min=1e-8)

        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, clip_range[0], clip_range[1])

        # --- 3. DDIM sampling ---
        pred_dir = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

        if prev_t >= 0:
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt((1.0 - alpha_bar_prev).clamp(min=0))

            x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * pred_dir

            if eta > 0:
                z = torch.randn_like(x_t)
                x_prev = x_prev + sigma_t * z
        else:
            x_prev = x_t

        return x_prev
