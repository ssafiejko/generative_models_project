import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from unet import CatDiffusion
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Union
from tqdm import tqdm
import uuid

def tensor_to_image(tensor: torch.Tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]  # take first image in batch if needed
    tensor = tensor.detach().cpu().clamp(0, 1)
    return to_pil_image(tensor)
    
@torch.no_grad()
def visualize_denoising(model: CatDiffusion, steps_to_show=10, device="cuda"):
    model.eval().to(device)
    cfg = model.cfg
    model.scheduler.set_timesteps(cfg.timesteps, device=device)

    x = torch.randn(1, cfg.channels, cfg.image_size, cfg.image_size, device=device)
    imgs = []

    step_indices = torch.linspace(0, len(model.scheduler.timesteps) - 1, steps_to_show, dtype=torch.long)
    timesteps_to_show = [model.scheduler.timesteps[i.item()] for i in step_indices]

    for t in tqdm(model.scheduler.timesteps, desc="Sampling", leave=False):
        with torch.autocast(device_type=device if device != "cpu" else "cpu", enabled=(device != "cpu")):
            noise_pred = model.unet(x, t).sample
        x = model.scheduler.step(noise_pred, t, x).prev_sample

        if t in timesteps_to_show:
            img = (x.clamp(-1, 1) + 1) / 2  # Scale to [0,1]
            imgs.append(img.cpu())

    grid = make_grid(torch.cat(imgs, dim=0), nrow=steps_to_show, padding=2)
    plt.figure(figsize=(steps_to_show * 1.5, 2))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Denoising Progression ({steps_to_show} Steps)")
    uuid_str = str(uuid.uuid4())
    plt.savefig(f'denoising_{uuid_str}')
    plt.show()


def compute_fid(
    model: CatDiffusion,
    real_loader,
    *,
    device: str | torch.device | None = None,
    num_samples: int = 1_000,
    gen_batch: int = 500,
) -> float:

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # --------------------------- REAL -----------------------------------
    added = 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = ((imgs + 1) / 2).clamp(0, 1)           
        imgs_u8 = (imgs * 255).round().to(torch.uint8) 
        fid.update(imgs_u8.to(device), real=True)
        added += imgs.size(0)
        if added >= num_samples:
            break

    # --------------------------- FAKE -----------------------------------
    model.eval()
    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            n = min(gen_batch, remaining)
            fake = model.sample(n, device=device)           
            fake_u8 = (fake * 255).round().to(torch.uint8)
            fid.update(fake_u8, real=False)
            remaining -= n

    return fid.compute().item()
