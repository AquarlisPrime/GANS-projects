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
import copy

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


class PerformerSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_head=64):
        super(PerformerSelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head

        self.qkv_proj = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

        self.attn = SelfAttention(
            dim=self.inner_dim,
            heads=num_heads,
            dim_head=dim_head,
            causal=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_proj(x).view(B, self.inner_dim * 3, H * W).permute(0, 2, 1)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self.attn(q, k, v)
        out = out.permute(0, 2, 1).view(B, self.inner_dim, H, W)
        out = self.out_proj(out)
        return out + x


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Changed to BatchNorm
        self.bn2 = nn.BatchNorm2d(out_channels)  # Changed to BatchNorm

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
        self.project = nn.Linear(latent_dim, 512 * 4 * 4)

        self.blocks = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True)),  # Changed to BatchNorm
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
        B = z.size(0)
        if z.dim() == 4:
            z = z.view(B, -1)

        x = self.project(z).view(B, 512, 4, 4)
        for i in range(step + 1):
            x = self.blocks[i](x)
        x = self.to_rgb[step](x)
        return x


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Changed to BatchNorm
        self.bn2 = nn.BatchNorm2d(out_channels)  # Changed to BatchNorm

    def forward(self, x):
        residual = self.downsample(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        return F.leaky_relu(out + residual, 0.2)


class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        mean_std = batch_std.mean().view(1, 1, 1, 1)
        shape = list(x.shape)
        shape[1] = 1
        std_feature = mean_std.expand(shape)
        return torch.cat([x, std_feature], 1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Helper to create blocks with optional attention
def _make_block(in_ch, out_ch, use_attention=False):
    if use_attention:
        return nn.Sequential(
            ResBlockDown(in_ch, out_ch),
            PerformerSelfAttention2D(out_ch)
        )
    else:
        return ResBlockDown(in_ch, out_ch)

class ProgressiveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [4, 8, 16, 32, 64, 128, 256, 512]

        # Selectively apply Performer attention at deeper layers
        self.blocks = nn.ModuleList([
            _make_block(channels[0], channels[1]),
            _make_block(channels[1], channels[2]),
            _make_block(channels[2], channels[3], use_attention=True),
            _make_block(channels[3], channels[4]),
            _make_block(channels[4], channels[5], use_attention=True),
            _make_block(channels[5], channels[6]),
            _make_block(channels[6], channels[7]),
        ])

        # From RGB to feature channels (with spectral norm)
        self.from_rgb = nn.ModuleList([
            spectral_norm(nn.Conv2d(1, c, kernel_size=1)) for c in channels
        ])

        # Final layers: Minibatch StdDev + final conv
        self.final = nn.Sequential(
            MinibatchStdDev(),
            spectral_norm(nn.Conv2d(513, 1, kernel_size=4)),
            nn.Flatten()
        )

    def forward(self, x, step, alpha):
        if step == 0:
            out = self.from_rgb[step](x)
            out = self.blocks[step](out)
        else:
            # Downscale previous resolution (skip if input is already very small)
            downscaled = F.avg_pool2d(x, 2) if x.size(2) > 1 and x.size(3) > 1 else x

            out_prev = self.from_rgb[step - 1](downscaled)
            out_prev = self.blocks[step - 1](out_prev)

            out = self.from_rgb[step](x)
            out = self.blocks[step](out)

            out = fade_in(alpha, out_prev, out)

        # Continue through the remaining blocks
        for block in self.blocks[step + 1:]:
            out = block(out)

        return self.final(out)

        
!pip install -q wandb
import wandb
wandb.login()
wandb.init(project="progressive-gan", anonymous="allow")
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import cv2
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
from torch.amp import autocast
from torchvision.utils import make_grid

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Set hyperparameters
latent_dim = 128
batch_size = 64
learning_rate = 1e-4
ema_decay = 0.999
fid_feature = 2048
image_size = 256  # Final resolution target

# ✅ Define CLAHE function
def apply_clahe(img_np):
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255.0
    img_np = img_np.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_np)

# ✅ Combined Dataset Class
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

        img_clahe = apply_clahe(img_np)
        img = Image.fromarray(img_clahe)

        if self.transform:
            img = self.transform(img)

        return img

# ✅ Define Transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# ✅ Dataset paths
root_dirs = [
    "/kaggle/input/struct-1-zip/particles/particles/images",
    "/kaggle/input/struct-1-zip/uhcs/uhcs/images"
]

# ✅ Load Dataset
dataset = CombinedMicrostructureDataset(root_dirs=root_dirs, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# ✅ Instantiate Models
model_G = ProgressiveGenerator(latent_dim=latent_dim).to(device)
model_D = ProgressiveDiscriminator().to(device)

# ✅ Optimizers
optimizer_G = optim.Adam(model_G.parameters(), lr=learning_rate, betas=(0.0, 0.99))
optimizer_D = optim.Adam(model_D.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# ✅ EMA
ema = EMA(model_G, decay=ema_decay)

# ✅ FID Metric
fid_metric = FrechetInceptionDistance(feature=fid_feature).to(device)

# ✅ AMP scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# ✅ wandb init
wandb.init(
    project="progressive-gan",
    name="progressive-performer-run-1",
    config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_dim": latent_dim,
        "ema_decay": ema_decay,
        "loss": "WGAN-GP",
        "attention": "Performer (Linear Attention)",
        "resolution_schedule": "4x4 to 256x256",
        "dataset": "CLAHE-enhanced microstructure images",
    }
)


from torchvision import transforms
from torch.utils.data import DataLoader

def get_combined_dataloader(res, batch_size):
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.CenterCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CombinedMicrostructureDataset(
        root_dirs=[
            "/kaggle/input/struct-1-zip/particles/particles/images",
            "/kaggle/input/struct-1-zip/uhcs/uhcs/images"
        ],
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def get_combined_dataloader(res, batch_size):
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.CenterCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CombinedMicrostructureDataset(
        root_dirs=[
            "/kaggle/input/struct-1-zip/particles/particles/images",
            "/kaggle/input/struct-1-zip/uhcs/uhcs/images"
        ],
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


# ✅ Training Hyperparams
total_steps = int(torch.log2(torch.tensor(image_size // 4))) + 1
epochs_per_step = {i: 10 if i < total_steps - 1 else 30 for i in range(total_steps)}  # more at higher res
fid_eval_interval = 1  # Evaluate every N epochs
samples_to_generate = 16
fixed_noise = torch.randn(samples_to_generate, latent_dim, 1, 1, device=device)

# ✅ Loss functions
def gradient_penalty(D, real, fake, step, alpha):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)
    mixed_scores = D(interpolated, step, alpha)

    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient = gradient.view(batch_size, -1)
    gp = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ✅ Training Loop
for step in range(total_steps):
    res = 4 * 2 ** step
    alpha = 0
    data_loader = get_combined_dataloader(res, batch_size=64)
    batches_per_epoch = len(data_loader)
    print(f"🔁 Step {step}/{total_steps-1} | Resolution: {res}x{res}")

    for epoch in range(epochs_per_step[step]):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs_per_step[step]}")
        
        # ✅ FIXED: unpack only real
        for i, real in enumerate(pbar):
            real = real.to(device)

            if real.shape[-1] != res:
                real = F.interpolate(real, size=(res, res), mode='bilinear', align_corners=False)

            alpha = min(1, (i + epoch * batches_per_epoch) / (epochs_per_step[step] * batches_per_epoch))

            z = torch.randn(real.size(0), latent_dim, 1, 1, device=device)

            # ✅ Train Discriminator
            with autocast(device_type="cuda"):
                fake = model_G(z, step, alpha).detach()
                real_score = model_D(real, step, alpha)
                fake_score = model_D(fake, step, alpha)
                gp = gradient_penalty(model_D, real, fake, step, alpha)

                loss_D = -(real_score.mean() - fake_score.mean()) + 10 * gp

            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # ✅ Train Generator
            z = torch.randn(batch_size, latent_dim, device=device)
            with autocast(device_type="cuda"):
                fake = model_G(z, step, alpha)
                fake_score = model_D(fake, step, alpha)
                loss_G = -fake_score.mean()

            optimizer_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # ✅ EMA Update
            ema.update(model_G)

            # ✅ Logging
            pbar.set_postfix({
                "loss_D": loss_D.item(),
                "loss_G": loss_G.item(),
                "alpha": round(alpha, 3),
                "res": f"{res}x{res}"
            })

        # ✅ Evaluate FID every N epochs
        if (epoch + 1) % fid_eval_interval == 0:
            model_G.eval()
            fid_metric.reset()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for _ in range(5):
                        z_fid = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                        fake_images = ema.shadow(z_fid, step, alpha)  # Update: EMA for fake image generation
                        fid_metric.update(fake_images, real=False)

                        try:
                            real_batch = next(iter(data_loader)).to(device)
                            real_batch = F.interpolate(real_batch, size=(res, res), mode='bilinear', align_corners=False)
                            fid_metric.update(real_batch, real=True)
                        except StopIteration:
                            break

            fid_score = fid_metric.compute().item()
            wandb.log({
                f"FID_{res}x{res}": fid_score,
                f"Loss_D_{res}": loss_D.item(),
                f"Loss_G_{res}": loss_G.item(),
                "Alpha": alpha,
                "Step": step,
                "Resolution": res
            })

            # ✅ Image Log
            with torch.no_grad():
                fake_samples = ema.shadow(fixed_noise, step, alpha)
                grid = make_grid(fake_samples, nrow=4, normalize=True)
                wandb.log({f"Samples_{res}x{res}": [wandb.Image(grid)]})

            model_G.train()

print("✅ Training complete.")

