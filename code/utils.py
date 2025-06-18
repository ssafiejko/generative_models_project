import os
import numpy as np
from scipy import linalg
from torchvision.models.inception import inception_v3
from torchvision.transforms import Normalize
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Transformacja do normalizacji wejścia dla InceptionV3 ---
inception_normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def compute_activation_statistics(dataloader, model, device):
    model.eval()
    features = []

    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)

            # Wymuszenie RGB (3 kanałów)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            # Skalowanie do rozmiaru Inception i konwersja typu
            imgs = nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            imgs = imgs.to(dtype=torch.float32)

            # Normalizacja
            imgs = inception_normalize(imgs)

            preds = model(imgs)
            features.append(preds.cpu().numpy())

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def load_or_compute_real_stats(stats_path, dataloader, model, device):
    if os.path.exists(stats_path):
        print(f"Loading cached real image statistics from {stats_path}")
        data = np.load(stats_path)
        return data['mu'], data['sigma']
    else:
        print("Computing real image statistics...")
        mu, sigma = compute_activation_statistics(dataloader, model, device)
        np.savez(stats_path, mu=mu, sigma=sigma)
        return mu, sigma

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_fid_score_unified(generator_type, model, real_loader, cfg, num_fake=1000):
    """
    generator_type: str, one of ['gan', 'vae', 'diffusion']
    model: generator or diffusion model depending on type
    real_loader: DataLoader with real images
    cfg: configuration with batch_size, latent_dim, image_size, device, etc.
    num_fake: number of fake images to generate
    """

    device = cfg.device

    # Inception model for feature extraction
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = nn.Identity()
    inception.eval()

    # 1. Get real image statistics (with cache)
    real_stats_path = getattr(cfg, "real_stats_path", "real_stats.npz")
    mu_real, sigma_real = load_or_compute_real_stats(real_stats_path, real_loader, inception, device)

    # 2. Generate fake images
    fake_images = []
    model.eval()

    with torch.no_grad():
        if generator_type == 'gan':
            for _ in tqdm(range(num_fake // cfg.batch_size + 1), desc="Generating fakes (GAN)"):
                z = torch.randn(cfg.batch_size, cfg.latent_dim, 1, 1).to(device)
                imgs = model(z)
                fake_images.append(imgs)

        elif generator_type == 'vae':
            for _ in tqdm(range(num_fake // cfg.batch_size + 1), desc="Generating fakes (VAE)"):
                z = torch.randn(cfg.batch_size, cfg.latent_dim).to(device)
                imgs = model.decode(z)
                fake_images.append(imgs)

        elif generator_type == 'diffusion':
            for _ in tqdm(range(num_fake // cfg.batch_size + 1), desc="Sampling (Diffusion)"):
                imgs = model.sample(shape=(cfg.batch_size, 3, cfg.image_size, cfg.image_size), device=device)
                imgs = (imgs + 1) / 2.0  # from [-1, 1] to [0, 1]
                fake_images.append(imgs)


        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    fake_images = torch.cat(fake_images, dim=0)[:num_fake].cpu()

    # 3. Prepare fake DataLoader
    fake_dataset = torch.utils.data.TensorDataset(fake_images, torch.zeros(len(fake_images)))
    fake_loader = DataLoader(fake_dataset, batch_size=cfg.fid_batch)

    # 4. Get fake image statistics
    mu_fake, sigma_fake = compute_activation_statistics(fake_loader, inception, device)

    # 5. Compute FID
    fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid
