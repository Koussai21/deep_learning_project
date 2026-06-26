"""
Convolutional Autoencoder (AE) and Variational Autoencoder (VAE)
for anomaly detection on chest X-rays.

Anomaly score:
  Both models are trained exclusively on normal (healthy) samples.
  At inference, a high reconstruction error indicates an out-of-distribution image.

AE  : score = pixel-wise MSE between input and reconstruction.
VAE : score = ELBO loss = reconstruction MSE + β * KL divergence.
        The KL term regularises the latent space and makes the score
        more sensitive to distributional shifts.

Usage:
  model = VAE()
  recon, mu, log_var = model(x)
  score = vae_loss(recon, x, mu, log_var)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Shared building blocks ────────────────────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, last: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        ]
        if last:
            layers.append(nn.Sigmoid())
        else:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ── Autoencoder ───────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """Maps (B,3,H,W) → (B, latent_dim)."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            DownBlock(3,   32),
            DownBlock(32,  64),
            DownBlock(64,  128),
            DownBlock(128, 256),
        )
        self.pool   = nn.AdaptiveAvgPool2d(4)
        self.fc     = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ConvDecoder(nn.Module):
    """Maps (B, latent_dim) → (B, 3, H, W)."""

    def __init__(self, latent_dim: int = config.LATENT_DIM, out_size: int = 64):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64,  32),
            UpBlock(32,  3, last=True),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        out = self.decoder(h)
        # Resize to expected resolution if needed
        if out.shape[-1] != self.out_size:
            out = F.interpolate(out, size=self.out_size, mode="bilinear", align_corners=False)
        return out


class ConvAE(nn.Module):
    """
    Convolutional Autoencoder.
    Training loss: MSE(x, x_hat).
    Anomaly score: per-sample mean squared reconstruction error.
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM, image_size: int = 64):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, out_size=image_size)

    def forward(self, x):
        z   = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @staticmethod
    def anomaly_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (B,)."""
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2, 3])


# ── Variational Autoencoder ───────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """Maps (B,3,H,W) → (B, latent_dim) mu and log_var."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            DownBlock(3,   32),
            DownBlock(32,  64),
            DownBlock(64,  128),
            DownBlock(128, 256),
        )
        self.pool   = nn.AdaptiveAvgPool2d(4)
        flat_dim    = 256 * 4 * 4
        self.fc_mu      = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_log_var(h)


class VAE(nn.Module):
    """
    Variational Autoencoder.
    Training loss: ELBO = MSE reconstruction + β * KL divergence.
    Anomaly score: reconstruction error alone or full ELBO.
    """

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        image_size: int = 64,
        beta: float = config.VAE_BETA,
    ):
        super().__init__()
        self.beta    = beta
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, out_size=image_size)

    def reparameterize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z     = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def elbo_loss(self, x, x_hat, mu, log_var) -> torch.Tensor:
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl, recon_loss, kl

    @staticmethod
    def anomaly_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (B,)."""
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2, 3])
