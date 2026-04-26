from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
import json
import os

from DDIM.ForwardProcess import ForwardDiffusion
from DDIM.NoisePredictor import NoisePredictor
from DDIM.ReverseProcess import ReverseDiffusion

LOG_PATH = os.path.join(os.path.dirname(__file__), "debug-f56ef3.log")


def _log(hypothesis_id, run_id, location, message, data):
    entry = {
        "sessionId": "f56ef3",
        "id": f"log_{int(data.get('timestamp', 0))}",
        "timestamp": data.get("timestamp", 0),
        "location": location,
        "message": message,
        "data": {k: v for k, v in data.items() if k != "timestamp"},
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def run_forward_process():
    TIMESTEPS = 200
    N_SAMPLES = 50

    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)

    data, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1)
    data = data[:, [0, 2]]

    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    x_cur = torch.tensor(data, dtype=torch.float32)

    frames = []
    history = [x_cur.numpy().copy()]

    for t in range(TIMESTEPS):
        fig, ax = plt.subplots(figsize=(6, 6))
        hist_np = np.array(history)
        for p_idx in range(N_SAMPLES):
                ax.plot(hist_np[:, p_idx, 0], hist_np[:, p_idx, 1],
                        c='gray', alpha=0.3, linewidth=1)

        ax.scatter(x_cur[:, 0], x_cur[:, 1], c='blue', alpha=0.8, s=20, zorder=5)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"Forward Process: t = {t}/{TIMESTEPS}")
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]
        frames.append(image)
        if t == 0:
            for _ in range(15):
                frames.append(image)
        plt.close()

        x_cur = forward_diffusion.q_step(x_cur, t)
        history.append(x_cur.numpy().copy())

    last_frame = frames[-1]
    for _ in range(15):
        frames.append(last_frame)
    imageio.mimsave('Plot/ddim_forward_traj.gif', frames, fps=10, loop=0)
    print("Saved Plot/ddim_forward_traj.gif")


def run_reverse_process_ddim():
    """DDIM reverse process for Swiss Roll."""
    import time
    ts_import = int(time.time() * 1000)

    TIMESTEPS = 200
    BATCH_SIZE = 40
    LR = 1e-3
    EPOCHS = 50000
    N_SAMPLES = 3000
    DDIM_STEPS = 50
    ETA = 0.0

    data, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1)
    data = data[:, [0, 2]]
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    dataset = torch.tensor(data, dtype=torch.float32)

    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    model = NoisePredictor(input_dim=2, time_dim=32)

    print(f"Training ({EPOCHS} epochs)...")
    model.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, forward_diffusion=forward_diffusion)

    # DDIM sampling: build step sequence
    step_size = TIMESTEPS // DDIM_STEPS  # 10
    ddim_timesteps = []
    cur = TIMESTEPS - 1  # start from 199
    while cur > 0:
        ddim_timesteps.append(cur)
        cur -= step_size
    if ddim_timesteps[-1] != 0:
        ddim_timesteps.append(0)

    frames = []
    N_GEN_SAMPLES = 1000
    x_cur = torch.randn(N_GEN_SAMPLES, 2)

    # Log step sequence for verification
    _log("H2_fixed", "ddim_step_0", "DDIM_Swiss_Roll.py:step_seq",
         "DDIM step sequence", {
             "timestamp": ts_import,
             "ddim_timesteps": ddim_timesteps,
             "step_size": step_size,
             "n_steps": len(ddim_timesteps),
         })

    print(f"DDIM Sampling: {len(ddim_timesteps)} steps (eta={ETA})...")

    with torch.no_grad():
        for idx, t in enumerate(ddim_timesteps):
            prev_t = t - step_size if t - step_size > 0 else 0

            # Log x_cur stats BEFORE sampling (H3: model prediction quality)
            x_cur_norm = float(x_cur.norm().item())
            x_cur_mean = float(x_cur.mean().item())
            x_cur_std = float(x_cur.std().item())

            # Compute model prediction and x0 for this timestep (diagnostic)
            t_tensor = torch.full((N_GEN_SAMPLES,), t, dtype=torch.long)
            eps_pred = model(x_cur, t_tensor)

            alphas = 1.0 - forward_diffusion.betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            ab_t = float(alpha_bar[t].item())
            sqrt_ab_t = np.sqrt(ab_t)
            sqrt_1m_ab_t = np.sqrt(1.0 - ab_t)
            x0_from_pred = (x_cur - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t

            x0_mean = float(x0_from_pred.mean().item())
            x0_std = float(x0_from_pred.std().item())
            eps_norm = float(eps_pred.norm().item())

            _log("H2_fixed", f"ddim_step_{idx}", "DDIM_Swiss_Roll.py:before_sample",
                 f"Before DDIM step t={t}->prev_t={prev_t}", {
                     "timestamp": ts_import,
                     "step": idx,
                     "t": t,
                     "prev_t": prev_t,
                     "x_cur_norm": x_cur_norm,
                     "x_cur_mean": x_cur_mean,
                     "x_cur_std": x_cur_std,
                     "eps_pred_norm": eps_norm,
                     "x0_mean": x0_mean,
                     "x0_std": x0_std,
                     "alpha_bar_t": ab_t,
                 })

            x_cur = ReverseDiffusion.ddim_sample(
                model, x_cur, t, prev_t,
                forward_diffusion.betas, eta=ETA
            )

            # Log x_cur stats AFTER sampling (H1/H2: numerical stability)
            x_next_norm = float(x_cur.norm().item())
            x_next_mean = float(x_cur.mean().item())
            x_next_std = float(x_cur.std().item())

            _log("H2_fixed", f"ddim_step_{idx}", "DDIM_Swiss_Roll.py:after_sample",
                 f"After DDIM step t={t}->prev_t={prev_t}", {
                     "timestamp": ts_import,
                     "step": idx,
                     "t": t,
                     "prev_t": prev_t,
                     "x_next_norm": x_next_norm,
                     "x_next_mean": x_next_mean,
                     "x_next_std": x_next_std,
                     "norm_change": x_next_norm - x_cur_norm,
                 })

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x_cur[:, 0], x_cur[:, 1], c='blue', alpha=0.8, s=20, zorder=5)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"DDIM Reverse Process: step {idx+1}/{len(ddim_timesteps)}, t={t} -> t-1={prev_t}")
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(image[:, :, :3])
            plt.close()
            print(f"  DDIM step {idx+1}/{len(ddim_timesteps)}, t={t} -> t-1={prev_t}")

    last_frame = frames[-1]
    for _ in range(15):
        frames.append(last_frame)
    imageio.mimsave('Plot/ddim_reverse_process.gif', frames, fps=10, loop=0)
    print("Saved Plot/ddim_reverse_process.gif (DDIM, eta=0)")

    _log("H2_fixed", "post", "DDIM_Swiss_Roll.py:final",
         "DDIM sampling complete", {
             "timestamp": ts_import,
             "final_x_norm": float(x_cur.norm().item()),
             "final_x_mean": float(x_cur.mean().item()),
             "final_x_std": float(x_cur.std().item()),
         })


def run_reverse_process():
    """Original DDPM reverse process (kept for comparison)."""
    TIMESTEPS = 200
    BATCH_SIZE = 40
    LR = 1e-3
    EPOCHS = 20000
    N_SAMPLES = 3000

    data, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1)
    data = data[:, [0, 2]]
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    dataset = torch.tensor(data, dtype=torch.float32)

    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    model = NoisePredictor(input_dim=2, time_dim=32)

    model.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, forward_diffusion=forward_diffusion)

    frames = []
    N_GEN_SAMPLES = 1000
    x_cur = torch.randn(N_GEN_SAMPLES, 2)

    with torch.no_grad():
        for t in reversed(range(TIMESTEPS)):
            x_cur = ReverseDiffusion.p_sample(model, x_cur, t, forward_diffusion.betas, eta=0.0, clip_denoised=False)
            if t % 5 != 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x_cur[:, 0], x_cur[:, 1], c='blue', alpha=0.8, s=20, zorder=5)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"Reverse Process: t = {t}/{TIMESTEPS}")
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(image[:, :, :3])
            plt.close()

    last_frame = frames[-1]
    for _ in range(15):
        frames.append(last_frame)
    imageio.mimsave('Plot/ddim_reverse_process.gif', frames, fps=10, loop=0)
    print("Saved Plot/ddim_reverse_process.gif")

    pass


if __name__ == "__main__":
    run_forward_process()
    run_reverse_process_ddim()
