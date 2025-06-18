import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),          
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),        
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),       
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256*8*8, latent_dim)
        self.fc_logvar = nn.Linear(256*8*8, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 256*8*8)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),      
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        
        z = self.reparameterize(mu, logvar)
        
        x_rec = self.decoder_input(z)
        x_rec = self.decoder(x_rec)
        return x_rec, mu, logvar
    
    def decode(self, z):
        x_rec = self.decoder_input(z)
        x_rec = self.decoder(x_rec)
        return x_rec
    
    def generate(self, num_samples, device):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)

def train_vae(dataloader, cfg, hyperparams={}, save_path="vae_history.pkl", loss_type='mse'):
    model = VAE(cfg.latent_dim).to(cfg.device)
    beta = hyperparams.get('beta', 1.0)
    lr = hyperparams.get('lr', cfg.lr)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(cfg.beta1, 0.999))
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'loss_type': loss_type
    }
    
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        num_samples = 0
        
        for imgs, _ in dataloader:
            imgs = imgs.to(cfg.device)
            batch_size = imgs.size(0)
            
            recon_imgs, mu, logvar = model(imgs)
            
            if loss_type == 'mse':
                recon = F.mse_loss(recon_imgs, imgs, reduction='sum')
            elif loss_type == 'l1':
                recon = F.l1_loss(recon_imgs, imgs, reduction='sum')
            elif loss_type == 'bce':
                recon = F.binary_cross_entropy(
                    (recon_imgs + 1) / 2, 
                    (imgs + 1) / 2, 
                    reduction='sum'
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon + beta*kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()
            num_samples += batch_size
        
        avg_total = total_loss / num_samples
        avg_recon = recon_loss / num_samples
        avg_kl = kl_loss / num_samples
        
        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
              f"Total loss: {avg_total:.4f} "
              f"(Recon: {avg_recon:.4f} KL: {avg_kl:.4f}) "
              f"Loss type: {loss_type}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nHistoria VAE zapisana do: {save_path}")
    
    return model, history

def generate_vae_samples(model, num_samples, cfg):
    with torch.no_grad():
        z = torch.randn(num_samples, cfg.latent_dim).to(cfg.device)
        samples = model.decode(z)
    return samples

def vae_interpolate(model, z1, z2, cfg, steps=10):
    alphas = torch.linspace(0, 1, steps).to(cfg.device)
    zs = torch.stack([(1-a)*z1 + a*z2 for a in alphas])
    
    with torch.no_grad():
        interpolated = model.decode(zs)
    return interpolated

def load_vae_history(path):
    with open(path, 'rb') as f:
        history = pickle.load(f)
    return history