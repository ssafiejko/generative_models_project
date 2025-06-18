from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

try:
    from diffusers import DDPMScheduler, UNet2DModel
except ImportError as e:
    raise ImportError(
        "diffusers is required: pip install diffusers"
    ) from e


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class DiffusionConfig:
    """Hyper‑parameters for diffusion training."""

    image_size: int = 64  # input resolution (square)
    channels: int = 3  # RGB
    timesteps: int = 500  # diffusion steps
    lr: float = 1.0e-4  # learning rate
    weight_decay: float = 1.0e-2  # Adam weight decay
    epochs: int = 20  # training epochs


# -----------------------------------------------------------------------------
# Core diffusion wrapper holding UNet + scheduler
# -----------------------------------------------------------------------------


class CatDiffusion(nn.Module):
    """Container that holds the *denoiser* UNet and DDPM scheduler."""

    scheduler: DDPMScheduler 
    unet: UNet2DModel  

    def __init__(self, cfg: DiffusionConfig, lpb=3):
        super().__init__()
        self.cfg = cfg

        # --- Model -------------------------------------------------
        self.unet = UNet2DModel(
            sample_size=cfg.image_size,  # 64×64
            in_channels=cfg.channels,
            out_channels=cfg.channels,
            layers_per_block=lpb,
            block_out_channels=(32, 64, 128, 256),  # progressively doubled
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        # --- Beta schedule --------------------------------------------------------
        self.scheduler = DDPMScheduler(
            num_train_timesteps=cfg.timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",  # predict the noise
        )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n: int = 16, device: str | torch.device | None = None) -> torch.Tensor: 
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(device).eval()

        imgs = torch.randn(n, self.cfg.channels, self.cfg.image_size, self.cfg.image_size, device=device)
        self.scheduler.set_timesteps(self.cfg.timesteps, device=device)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            with torch.autocast(device.type if device.type != "cpu" else "cpu", enabled=(device.type != "cpu")):
                noise_pred = self.unet(imgs, t).sample
            imgs = self.scheduler.step(noise_pred, t, imgs).prev_sample

        return (imgs.clamp(-1, 1) + 1) / 2


# -----------------------------------------------------------------------------
# Training utility (public entry point)
# -----------------------------------------------------------------------------


def train_model(
    data_loader: Iterable[torch.Tensor],
    *,
    epochs: int = 20,
    lr: float = 1.0e-4,
    timesteps: int = 1_000,
    device: Optional[str | torch.device] = None,
    weight_decay: float = 1.0e-2,
    beta1: float = 0.9,
    loss_function = nn.MSELoss
) -> CatDiffusion:

    betas = (beta1, 0.999)

    cfg = DiffusionConfig(timesteps=timesteps, lr=lr, epochs=epochs, weight_decay=weight_decay)
    model = CatDiffusion(cfg)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    loss_eval = loss_function()
    loss_list = []
    opt = optim.Adam(model.parameters(), lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs")
    # betas      = model.scheduler.betas.to(device)
    # alphacum   = model.scheduler.alphas_cumprod.to(device)
    for _ in epoch_bar:
        running_loss = 0.0
        batch_bar = tqdm(data_loader, desc="Batches", leave=False)
        for im in batch_bar:
            x = im[0].to(device)
            batch_size = x.size(0)
        
            opt.zero_grad(set_to_none=True)
        
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=device)  # dtype long by default
            noise = torch.randn_like(x)
            noisy_x = model.scheduler.add_noise(x, noise, t)
        
            # beta_t      = betas[t][:, None, None, None]
            # alpha_cum_t = alphacum[t][:, None, None, None]
        
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred_noise = model.unet(noisy_x, t).sample
                loss = loss_eval(pred_noise, noise)
        
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * batch_size
            batch_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
        avg = running_loss / len(data_loader.dataset)
        loss_list.append(avg)
        epoch_bar.set_postfix({"epoch_mse": f"{avg:.4f}"})

    return model, loss_list
