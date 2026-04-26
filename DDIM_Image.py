import matplotlib.image as img
import torch
import time
import imageio
import numpy as np

from DDIM.ForwardProcess import ForwardDiffusion
from DDIM.ReverseProcess import ReverseDiffusion
from DDIM.NoisePredictor import DiffUNet
from DDIM.NoisePredictor import EMA
from Dataset import OxfordPetLoader


def tensor_to_image(x_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    x_tensor = x_tensor.clone()
    x_np = (x_tensor + 1.0) / 2.0
    x_np = x_np.clamp(0.0, 1.0).cpu().numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))
    x_np = (x_np * 255.0).round().astype(np.uint8)
    return x_np


def ddim_sample(model, forward_diffusion, device, timesteps=1000, ddim_steps=50, eta=0.0, save_path='Plot/ddim_cat_reverse_process.gif'):
    step_size = timesteps // ddim_steps
    ddim_timesteps = []
    cur = timesteps - 1
    while cur > 0:
        ddim_timesteps.append(cur)
        cur -= step_size
    if ddim_timesteps[-1] != 0:
        ddim_timesteps.append(0)

    frames = []
    x_cur = torch.randn(1, 3, 256, 256).to(device)

    print(f"DDIM Sampling: {len(ddim_timesteps)} steps (eta={eta})...")

    with torch.no_grad():
        for idx, t in enumerate(ddim_timesteps):
            prev_t = t - step_size if t - step_size > 0 else 0
            x_cur = ReverseDiffusion.ddim_sample(
                model, x_cur, t, prev_t,
                forward_diffusion.betas, eta=eta
            )

            if t % 20 == 0 or idx == len(ddim_timesteps) - 1:
                x_images = tensor_to_image(x_cur)
                frames.append(x_images[0])
                print(f"  DDIM step {idx+1}/{len(ddim_timesteps)}, t={t} -> t-1={prev_t}")

            if idx == len(ddim_timesteps) - 1:
                for _ in range(30):
                    frames.append(x_images[0])

    imageio.mimsave(save_path, frames, fps=30, loop=0)
    print(f"Saved {save_path} (DDIM, {ddim_steps} steps, eta={eta})")


def run_reverse_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTEPS = 1000
    N_IMAGE = 3
    BATCH_SIZE = 3
    LR = 1e-3
    EPOCHS = 100000
    if device.type == 'cuda':
        print(f"  >> GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  >> cuDNN Version: {torch.backends.cudnn.version()}")

    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    forward_diffusion.to(device)

    print(f"Loading Oxford-IIIT Pet Dataset (Cats) - Limited to {N_IMAGE} images...")
    data_loader = OxfordPetLoader(root='./data', batch_size=N_IMAGE, download=True, cat_only=True).get_loader()

    imgs, _ = next(iter(data_loader))
    imgs = imgs[:N_IMAGE].to(device)

    model = DiffUNet(input_channels=3, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    ema_model = EMA(model, beta=0.995)

    print(f"Start Training ({EPOCHS} steps)...")
    for epoch in range(EPOCHS):
        time_start = time.time()
        total_loss = 0.0

        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=device)

        noise = torch.randn_like(imgs)
        x_t = forward_diffusion.q_sample(imgs, t, noise=noise)

        predicted_noise = model(x_t, t)

        loss = loss_function(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema_model.update(model)
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / N_IMAGE
        time_end = time.time()
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Time: {time_end - time_start:.2f}s")

    print("Training Completed.")

    # DDIM sampling: 50 steps instead of 1000, eta=0 (deterministic)
    ddim_sample(
        model=ema_model.ema_model,
        forward_diffusion=forward_diffusion,
        device=device,
        timesteps=TIMESTEPS,
        ddim_steps=100,
        eta=0.0,
        save_path='Plot/ddim_cat_reverse_process.gif'
    )


if __name__ == "__main__":
    run_reverse_process()
