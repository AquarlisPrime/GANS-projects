import os

dataset_path = "/kaggle/input/struct-1-zip"
for root, dirs, files in os.walk(dataset_path):
    print(f"📂 {root}")
    for file in files:
        print(f"  📄 {file}")
import os
import torch
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# ---- 1. CLAHE enhancement ----
def apply_clahe(img_np):
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255.0
    img_np = img_np.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_np)

# ---- 2. Combined Dataset Class ----
class CombinedMicrostructureDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.image_paths = []
        for dir_path in root_dirs:
            self.image_paths += [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".png", ".tif"))
            ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        ext = os.path.splitext(path)[1].lower()

        if ext == ".tif":
            img_np = tiff.imread(path)
        else:
            img_np = np.array(Image.open(path).convert("L"))

        # Apply CLAHE
        img_clahe = apply_clahe(img_np)

        # Convert to PIL
        img = Image.fromarray(img_clahe)

        if self.transform:
            img = self.transform(img)

        return img

# ---- 3. Data Transform & Loader ----
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Paths to both image folders
d_particles = "/kaggle/input/struct-1-zip/particles/particles/images"
d_uhcs = "/kaggle/input/struct-1-zip/uhcs/uhcs/images"

# Load combined dataset
root_dirs = [d_particles, d_uhcs]
dataset = CombinedMicrostructureDataset(root_dirs=root_dirs, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
# Paths for particles and UHCS
d_particles = "/kaggle/input/struct-1-zip/particles/particles/images"
l_particles = "/kaggle/input/struct-1-zip/particles/particles/labels"
d_uhcs = "/kaggle/input/struct-1-zip/uhcs/uhcs/images"
l_uhcs = "/kaggle/input/struct-1-zip/uhcs/uhcs/labels"

import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Get a batch of images
real_batch = next(iter(dataloader))

# Unnormalize the images for display
def unnormalize(img):
    img = img * 0.5 + 0.5  # Undo normalization
    return img

# Make a grid and convert to numpy for plotting
grid_img = vutils.make_grid(unnormalize(real_batch[:64]), nrow=8, padding=2)
np_grid = grid_img.permute(1, 2, 0).cpu().numpy()

# Show the image grid
plt.figure(figsize=(10, 10))
plt.imshow(np_grid, cmap="gray")
plt.axis("off")
plt.title("Sample CLAHE + Augmented Microstructure Images")
plt.show()

!pip install performer-pytorch
# 4. Loss Functions and Gradient Penalty (Progressive-Compatible)
def compute_gradient_penalty(D, real_samples, fake_samples, step, alpha, lambda_gp=10):
    alpha_interp = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha_interp * real_samples + (1 - alpha_interp) * fake_samples).requires_grad_(True)
    
    d_interpolates = D(interpolates, step, alpha)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty

def generator_loss(fake_scores):
    """WGAN-GP Generator loss"""
    return -fake_scores.mean()

def discriminator_loss(real_scores, fake_scores, gradient_penalty):
    """WGAN-GP Discriminator loss"""
    return fake_scores.mean() - real_scores.mean() + gradient_penalty
!pip install torch-fidelity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from performer_pytorch import SelfAttention


class PerformerSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_head=64):
        super(PerformerSelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head

        self.qkv_proj = nn.Linear(in_channels, self.inner_dim * 3)
        self.out_proj = nn.Linear(self.inner_dim, in_channels)

        self.attn = SelfAttention(
            dim=self.inner_dim,
            heads=num_heads,
            dim_head=dim_head,
            causal=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        qkv = self.qkv_proj(x_flat)                   # (B, N, 3 * inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)                # (B, N, inner_dim)

        out = self.attn(q, k, v)                      # (B, N, inner_dim)
        out = self.out_proj(out)                      # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + x  # Residual


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.upsample(x)
        out = F.relu(self.bn1(self.conv1(F.interpolate(x, scale_factor=2))))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


def fade_in(alpha, upscaled, generated):
    return alpha * generated + (1 - alpha) * upscaled


class ProgressiveGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(latent_dim, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)),
            ResBlockUp(512, 256),
            nn.Sequential(ResBlockUp(256, 128), PerformerSelfAttention2D(128)),
            ResBlockUp(128, 64),
            nn.Sequential(ResBlockUp(64, 32), PerformerSelfAttention2D(32)),
            ResBlockUp(32, 16),
            ResBlockUp(16, 8),
            ResBlockUp(8, 4),
        ])
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1) for c in [512, 256, 128, 64, 32, 16, 8, 4]
        ])

    def forward(self, z, step, alpha):
        x = z
        for i in range(step + 1):
            x = self.blocks[i](x)
        if step == 0:
            return torch.tanh(self.to_rgb[step](x))
        upscaled = F.interpolate(self.to_rgb[step - 1](self.blocks[step - 1](z)), scale_factor=2)
        out = self.to_rgb[step](x)
        return torch.tanh(fade_in(alpha, upscaled, out))


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        return F.relu(out + residual)


class ProgressiveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [4, 8, 16, 32, 64, 128, 256, 512]
        self.blocks = nn.ModuleList([
            ResBlockDown(channels[0], channels[1]),
            ResBlockDown(channels[1], channels[2]),
            nn.Sequential(ResBlockDown(channels[2], channels[3]), PerformerSelfAttention2D(channels[3])),
            ResBlockDown(channels[3], channels[4]),
            nn.Sequential(ResBlockDown(channels[4], channels[5]), PerformerSelfAttention2D(channels[5])),
            ResBlockDown(channels[5], channels[6]),
            ResBlockDown(channels[6], channels[7]),
        ])
        self.from_rgb = nn.ModuleList([
            nn.Conv2d(1, c, kernel_size=1) for c in channels
        ])
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4)),
            nn.Flatten()
        )

    def forward(self, x, step, alpha):
        if step == 0:
            out = self.from_rgb[step](x)
            out = self.blocks[step](out)
        else:
            downscaled = F.avg_pool2d(x, 2)
            out_prev = self.from_rgb[step - 1](downscaled)
            out = self.from_rgb[step](x)
            out = fade_in(alpha, out_prev, out)
            out = self.blocks[step](out)

        for i in range(step + 1, len(self.blocks)):
            out = self.blocks[i](out)

        return self.final(out)
