# Generative Models Repository

![cats](/example_images/interpolated_dcgan_cats1.png) 

*Example generated images from our models*

This repository contains PyTorch implementations of modern generative models, with extensive experiments on cat image generation.

##  Implemented Models

###  GAN Variants
| Model       | Key Features                          | Training File      |
|-------------|---------------------------------------|--------------------|
| DCGAN       | Basic GAN with BCE loss               | `gan.py`           |
| LSGAN       | Least Squares loss for stability      | `gan.py`           |
| SNGAN       | Spectral Normalization + Hinge loss   | `gan.py`           |

###  VAE
- Configurable reconstruction loss (MSE/L1/BCE)
- β-VAE support for disentanglement

###  Diffusion
- UNet architecture with residual blocks
- DDPM scheduler
- Squared cosine beta schedule

##  Key Features

- **Training Utilities**  
  Unified training loops for all models with history tracking

- **Evaluation**  
  FID score computation with InceptionV3

- **Visualization**  
  Latent space interpolation functions

- **Flexible Configs**  
  Dataclass-based hyperparameter management


## Example Usage

## 🛠 Example Usage

### 1. Training Models

#### GAN Training (DCGAN/LSGAN/SNGAN)
```python
from gan import train_dcgan, train_lsgan, train_sngan
from config import Config

# Initialize config
cfg = Config(
    latent_dim=100,
    image_size=64,
    num_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Choose your GAN variant
generator, discriminator, history = train_sngan(
    dataloader=your_dataloader,
    cfg=cfg,
    hyperparams={'lr': 0.0002, 'betas': (0.5, 0.999)},
    save_path="training_history.pkl"
)
```
#### VAE Training 
```python
from vae import train_vae

vae_model, history = train_vae(
    dataloader=your_dataloader,
    cfg=cfg,
    hyperparams={'beta': 0.5, 'lr': 0.001},
    loss_type='mse'  # 'mse'|'l1'|'bce'
)
```
#### Diffusion Model Training
```python
from unet import train_model

diffusion_model, losses = train_model(
    data_loader=your_dataloader,
    epochs=50,
    lr=1e-4,
    timesteps=1000,
    device='cuda'
)
```
### 2. Generating samples
```python
# GAN/VAE generation
num_samples = 16
z = torch.randn(num_samples, cfg.latent_dim, 1, 1).to(cfg.device)  # For GAN
# z = torch.randn(num_samples, cfg.latent_dim).to(cfg.device)  # For VAE

with torch.no_grad():
    samples = generator(z)  # For GAN
    # samples = vae_model.decode(z)  # For VAE

# Diffusion sampling
samples = diffusion_model.sample(n=16, device=cfg.device)
```
### 3. Latent space interpolation
```python
# For GAN/VAE
z1 = torch.randn(1, cfg.latent_dim, 1, 1).to(cfg.device)
z2 = torch.randn(1, cfg.latent_dim, 1, 1).to(cfg.device)

interpolated = interpolate_and_generate(
    generator, 
    z1, z2, 
    cfg, 
    steps=10
)

# For VAE specifically
interpolated = vae_interpolate(
    vae_model,
    z1.squeeze(),
    z2.squeeze(),
    cfg
)
```
### 4. Evaluation
```python
from utils import compute_fid_score_unified

# Compute FID for any model type
fid_score = compute_fid_score_unified(
    generator_type='gan',  # 'gan'|'vae'|'diffusion'
    model=generator,
    real_loader=validation_loader,
    cfg=cfg,
    num_fake=5000
)
print(f"FID Score: {fid_score:.2f}")
```
