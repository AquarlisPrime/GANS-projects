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
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
# 1. Dataset with Feature Engineering (CLAHE)
def apply_clahe(img):
    # Normalize to [0, 255] and convert to uint8
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# Updated Dataset with CLAHE
class MicrostructureDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.img_paths[idx])  # Load grayscale TIFF

        # Apply CLAHE feature engineering
        img = apply_clahe(img)

        # Convert NumPy array to PIL image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img
# 2. Transform and Dataloader
dataset_path = "/kaggle/input/struct-1-zip/particles/particles/images"
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(256),  # or 256
    transforms.CenterCrop(256),  # or 256
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MicrostructureDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Paths for particles and UHCS
d_particles = "/kaggle/input/struct-1-zip/particles/particles/images"
l_particles = "/kaggle/input/struct-1-zip/particles/particles/labels"
d_uhcs = "/kaggle/input/struct-1-zip/uhcs/uhcs/images"
l_uhcs = "/kaggle/input/struct-1-zip/uhcs/uhcs/labels"
# 3. Generator and Discriminator
import torch.nn.utils.spectral_norm as spectral_norm

class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockUp, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.upsample(x) + self.conv_path(F.interpolate(x, scale_factor=2))

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0),  # -> 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )

        self.upsample_blocks = nn.Sequential(
            ResBlockUp(1024, 512),   # -> 8x8
            ResBlockUp(512, 256),    # -> 16x16
            ResBlockUp(256, 128),    # -> 32x32
            ResBlockUp(128, 64),     # -> 64x64
            ResBlockUp(64, 32),      # -> 128x128
            ResBlockUp(32, 16),      # -> 256x256
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.init(z)
        x = self.upsample_blocks(x)
        return self.final(x)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: 1 x 256 x 256
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),     # -> 32 x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),    # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),   # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # -> 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), # -> 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),   # -> 1 x 1 x 1
        )

    def forward(self, img):
        return self.model(img).view(-1)
# 4. Loss Functions and Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generator_loss(fake_scores):
    return -fake_scores.mean()

def discriminator_loss(real_scores, fake_scores, gradient_penalty, lambda_gp=10):
    return fake_scores.mean() - real_scores.mean() + lambda_gp * gradient_penalty
import os
from PIL import Image
from pathlib import Path
import shutil

# Source (read-only) and destination (writable) directories
src_dir = Path('/kaggle/input/struct-1-zip/uhcs/uhcs/images')
dst_dir = Path('/kaggle/working/uhcs_converted_images')
dst_dir.mkdir(parents=True, exist_ok=True)

# Convert .tif to .png
for i, fname in enumerate(os.listdir(src_dir)):
    if fname.endswith('.tif'):
        src_path = src_dir / fname
        img = Image.open(src_path).convert("RGB")
        img.save(dst_dir / f"{i:05d}.png")
import torch.nn as nn

# 🔧 Weight Initialization Function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
!pip install torch-fidelity
import os
import torch
import pandas as pd
from PIL import Image
import torchvision.utils as vutils
from torch_fidelity import calculate_metrics
from IPython.display import clear_output
import matplotlib.pyplot as plt
import shutil

# Training Parameters
latent_dim = 100
n_epochs = 1000
n_critic = 5
save_every = 10  # Save checkpoints + FID every N epochs

# Initialize models
gen = Generator(latent_dim).apply(weights_init)
disc = Discriminator().apply(weights_init)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen, disc = gen.to(device), disc.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.0, 0.9))

# Folders
os.makedirs("generated_progress", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("fid_generated", exist_ok=True)

# Fixed latent vector for consistent visualization
fixed_z = torch.randn(64, latent_dim, 1, 1).to(device)

# FID CSV
fid_scores = []
fid_csv_path = "fid_scores.csv"

for epoch in range(n_epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # === Train Discriminator ===
        for _ in range(n_critic):
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
            fake_imgs = gen(z).detach()
            real_scores = disc(real_imgs)
            fake_scores = disc(fake_imgs)

            gp = compute_gradient_penalty(disc, real_imgs, fake_imgs)
            d_loss = discriminator_loss(real_scores, fake_scores, gp)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        # === Train Generator ===
        z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
        fake_imgs = gen(z)
        fake_scores = disc(fake_imgs)
        g_loss = generator_loss(fake_scores)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    # === Save Sample Images Every Epoch ===
    gen.eval()
    with torch.no_grad():
        sample_imgs = gen(fixed_z)
        vutils.save_image(sample_imgs, f"generated_progress/epoch_{epoch+1:03d}.png", nrow=8, normalize=True)
    gen.train()

    clear_output(wait=True)
    print(f"[Epoch {epoch+1}/{n_epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # === Checkpointing + FID Every `save_every` Epochs ===
    if (epoch + 1) % save_every == 0:
        # Save model checkpoints
        torch.save(gen.state_dict(), f"checkpoints/gen_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"checkpoints/disc_epoch_{epoch+1}.pth")

        # Clean fid_generated folder
        for f in os.listdir("fid_generated"):
            os.remove(os.path.join("fid_generated", f))

        # Generate images for FID
        gen.eval()
        with torch.no_grad():
            z_fid = torch.randn(128, latent_dim, 1, 1).to(device)
            fid_imgs = gen(z_fid)
            fid_imgs = (fid_imgs + 1) / 2  # Scale from [-1, 1] to [0, 1]
            for idx, img in enumerate(fid_imgs):
                vutils.save_image(img, f"fid_generated/img_{epoch+1}_{idx}.png")

        # Calculate FID using torch-fidelity
        metrics = calculate_metrics(
            input1='fid_generated',
            input2='/kaggle/working/uhcs_converted_images',  # Real image dir (converted to PNGs beforehand)
            fid=True,
            verbose=False
        )

        fid_score = metrics['frechet_inception_distance']
        fid_scores.append((epoch + 1, fid_score))
        pd.DataFrame(fid_scores, columns=["Epoch", "FID"]).to_csv(fid_csv_path, index=False)

        print(f"📊 Epoch {epoch+1} | FID Score: {fid_score:.4f}")

# Final FID plot
df = pd.read_csv(fid_csv_path)
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["FID"], marker='o', linestyle='-', color='green')
plt.xlabel("Epoch")
plt.ylabel("FID Score")
plt.title("FID Score Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("fid_curve.png")
plt.show()

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
from PIL import Image
import os

last_image_path = "/kaggle/working/generated_progress/epoch_249.png"
img = Image.open(last_image_path)

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.axis("off")
plt.title("Generated Samples - Epoch 250")
plt.show()
print(real_imgs.shape, fake_imgs.shape)

import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Set model to eval mode
gen.eval()

# Generate a batch of images
z = torch.randn(64, latent_dim, 1, 1).to(device)
with torch.no_grad():
    fake_imgs = gen(z).detach().cpu()

# Plot a grid of generated images
grid = vutils.make_grid(fake_imgs, nrow=8, normalize=True)
plt.figure(figsize=(10,10))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Generated Synthetic Microstructure Images")
plt.show()
#save gen model
torch.save(gen.state_dict(), "wgan_gp_generator.pth")
print("Generator model saved as 'wgan_gp_generator.pth'")
!pip install imageio
import imageio
import os

# Directory where images were saved
image_dir = "generated_progress"
output_gif = "training_progress.gif"

# Get all saved images and sort by epoch number
image_files = sorted(
    [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
)

# Create and save GIF
images = [imageio.imread(f) for f in image_files]
imageio.mimsave(output_gif, images, fps=5)  # Adjust fps for speed

print(f"GIF saved as {output_gif}")
!pip install torchmetrics scikit-image
!pip install torch-fidelity --quiet
!pip install torchmetrics[image] --quiet

import os
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from PIL import Image
import torch
from tqdm import tqdm
import tifffile as tiff  # ✅ Added for .tif support

# FID expects 299x299 RGB images
fid_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Load real images using tifffile
real_dir = "/kaggle/input/struct-1-zip/particles/particles/images"
real_images = []
for fname in tqdm(os.listdir(real_dir)):
    if fname.endswith(".tif"):
        path = os.path.join(real_dir, fname)
        img = tiff.imread(path)              # ✅ Load as NumPy array
        img = Image.fromarray(img).convert("RGB")  # ✅ Convert to RGB
        img = fid_transform(img)
        real_images.append(img)

# Load generated images (assumed to be .png)
generated_dir = "/kaggle/working/generated_progress"
generated_images = []
for fname in tqdm(sorted(os.listdir(generated_dir))[-50:]):  # Use last 50 for stability
    if fname.endswith(".png"):
        img = Image.open(os.path.join(generated_dir, fname)).convert("RGB")
        img = fid_transform(img)
        generated_images.append(img)

# Stack tensors
real_tensor = torch.stack(real_images)
fake_tensor = torch.stack(generated_images)

# Calculate FID
fid = FrechetInceptionDistance(normalize=True)
fid.update(real_tensor, real=True)
fid.update(fake_tensor, real=False)
fid_score = fid.compute().item()

print(f"FID Score: {fid_score:.4f}")
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("generated_progress/epoch_249.png")
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.title("Generated Samples at Epoch 250")
plt.show()
