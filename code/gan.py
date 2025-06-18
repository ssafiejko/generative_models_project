import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.nn.utils import spectral_norm


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, img):
        return self.net(img).view(-1, 1).squeeze(1)



class SNGANGenerator(nn.Module):
    def __init__(self, latent_dim, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   3,     4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class SNGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*8, 1,   4, 1, 0, bias=False))
        )
    def forward(self, img):
        return self.net(img).view(-1)

def train_sngan(dataloader, cfg, hyperparams={}, save_path="sngan_history.pkl"):
    gen = SNGANGenerator(cfg.latent_dim, cfg.ngf).to(cfg.device)
    disc = SNGANDiscriminator(cfg.ndf).to(cfg.device)

    lr = hyperparams.get('lr', cfg.lr)
    betas = hyperparams.get('betas', (0.0, 0.9))

    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=betas)

    history = {'loss_d': [], 'loss_g': []}

    for epoch in range(cfg.num_epochs):
        epoch_loss_d, epoch_loss_g = 0.0, 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(cfg.device)
            bs = real_imgs.size(0)

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=cfg.device)
            fake_imgs = gen(z).detach()

            d_real = disc(real_imgs)
            d_fake = disc(fake_imgs)

            loss_d_real = torch.mean(torch.relu(1.0 - d_real))
            loss_d_fake = torch.mean(torch.relu(1.0 + d_fake))
            loss_d = loss_d_real + loss_d_fake

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=cfg.device)
            fake_imgs = gen(z)
            d_fake_for_g = disc(fake_imgs)
            loss_g = -torch.mean(d_fake_for_g)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            num_batches += 1

        avg_d = epoch_loss_d / num_batches
        avg_g = epoch_loss_g / num_batches
        history['loss_d'].append(avg_d)
        history['loss_g'].append(avg_g)

        print(f"[SNGAN] Epoch {epoch+1}/{cfg.num_epochs} - Loss D: {avg_d:.4f}, Loss G: {avg_g:.4f}")

    with open(save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Historia SNGAN zapisana do: {save_path}")

    return gen, disc, history


    
def compute_gradient_penalty(disc, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = disc(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_dcgan(dataloader, cfg, hyperparams={}, save_path="dcgan_history.pkl"):
    gen = DCGANGenerator(cfg.latent_dim, cfg.ngf).to(cfg.device)
    disc = DCGANDiscriminator(cfg.ndf).to(cfg.device)
    opt_g = optim.Adam(gen.parameters(), lr=hyperparams.get('lr', cfg.lr), betas=(cfg.beta1, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=hyperparams.get('lr', cfg.lr), betas=(cfg.beta1, 0.999))
    criterion = nn.BCELoss()

    history = {
        "loss_g": [],
        "loss_d": []
    }

    for epoch in range(cfg.num_epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(cfg.device)
            bs = real_imgs.size(0)

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=cfg.device)
            fake_imgs = gen(z)
            labels_real = torch.ones(bs, device=cfg.device)
            labels_fake = torch.zeros(bs, device=cfg.device)

            d_real = disc(real_imgs)
            d_fake = disc(fake_imgs.detach())

            loss_d = criterion(d_real, labels_real) + criterion(d_fake, labels_fake)
            disc.zero_grad()
            loss_d.backward()
            opt_d.step()

            loss_g = criterion(disc(fake_imgs), labels_real)
            gen.zero_grad()
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            num_batches += 1

        avg_loss_d = epoch_loss_d / num_batches
        avg_loss_g = epoch_loss_g / num_batches
        history["loss_d"].append(avg_loss_d)
        history["loss_g"].append(avg_loss_g)

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}]  Loss D: {avg_loss_d:.4f}, Loss G: {avg_loss_g:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(history, f)

    print(f"\n Historia treningu zapisana do: {save_path}")
    return gen, disc, history


def train_lsgan(dataloader, cfg, hyperparams={}, save_path="lsgan_history.pkl"):
    gen = DCGANGenerator(cfg.latent_dim, cfg.ngf).to(cfg.device)
    disc = DCGANDiscriminator(cfg.ndf).to(cfg.device)
    opt_g = optim.Adam(gen.parameters(), lr=hyperparams.get('lr', cfg.lr), betas=(cfg.beta1, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=hyperparams.get('lr', cfg.lr), betas=(cfg.beta1, 0.999))
    criterion = nn.MSELoss()

    history = {"loss_g": [], "loss_d": []}

    for epoch in range(cfg.num_epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(cfg.device)
            bs = real_imgs.size(0)

            z = torch.randn(bs, cfg.latent_dim, 1, 1, device=cfg.device)
            fake_imgs = gen(z)

            real_labels = torch.full((bs,), 0.9, device=cfg.device)  
            fake_labels = torch.full((bs,), 0.0, device=cfg.device)

            d_real = disc(real_imgs)
            d_fake = disc(fake_imgs.detach())

            loss_d = 0.5 * (criterion(d_real, real_labels) + criterion(d_fake, fake_labels))
            disc.zero_grad()
            loss_d.backward()
            opt_d.step()

            output = disc(fake_imgs)
            loss_g = 0.5 * criterion(output, real_labels)

            gen.zero_grad()
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            num_batches += 1

        history["loss_d"].append(epoch_loss_d / num_batches)
        history["loss_g"].append(epoch_loss_g / num_batches)
        print(f"[LSGAN] Epoch {epoch+1}/{cfg.num_epochs} - Loss D: {epoch_loss_d/num_batches:.4f}, Loss G: {epoch_loss_g/num_batches:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(history, f)

    print(f"\n Historia LSGAN zapisana do: {save_path}")
    return gen, disc, history



def interpolate_and_generate(gen, z1, z2, cfg, steps=10):
    alphas = np.linspace(0, 1, steps)
    zs = torch.stack([(1 - a) * torch.Tensor(z1) + a * torch.Tensor(z2) for a in alphas])
    zs = zs.view(steps, cfg.latent_dim, 1, 1).to(cfg.device)  
    
    with torch.no_grad():
        imgs = gen(zs)
    return imgs