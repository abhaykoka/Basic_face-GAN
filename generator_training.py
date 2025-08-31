import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ✅ Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

import pandas as pd

# ------------------------
# DCGAN Generator & Discriminator
# ------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*2, feature_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*4, feature_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


# ------------------------
# Training Loop
# ------------------------
def train(data_path, epochs, batch_size, z_dim, img_size, lr, device, out_dir):
    # ------------------------
    # Prepare dataset
    # ------------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(subdirs) == 0:
        print("⚠️ Flat dataset detected. Creating dummy 'profile' class.")
        profile_dir = os.path.join(data_path, "profile")
        os.makedirs(profile_dir, exist_ok=True)
        for f in os.listdir(data_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                old_path = os.path.join(data_path, f)
                new_path = os.path.join(profile_dir, f)
                os.rename(old_path, new_path)

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ------------------------
    # Models, Loss, Optimizers
    # ------------------------
    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # Metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    is_score = InceptionScore().to(device)
    # ✅ Reduced subset_size to avoid small batch errors
    kid = KernelInceptionDistance(subset_size=min(10, batch_size)).to(device)

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "evaluation_metrics.csv")
    metrics = []

    # ------------------------
    # Training Loop
    # ------------------------
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            real_label = torch.ones(b_size, device=device)
            fake_label = torch.zeros(b_size, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, real_label)

            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake_label)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_imgs)
            loss_G = criterion(output, real_label)
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

        # Save sample images
        if (epoch+1) % 5 == 0 or epoch == epochs-1:
            fake_samples = generator(fixed_noise).detach().cpu()
            save_image(fake_samples[:25], os.path.join(out_dir, f"epoch_{epoch+1}.png"), nrow=5, normalize=True)

            # ------------------------
            # Compute metrics (GPU + uint8 safe)
            # ------------------------
            real_resized = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)
            fake_resized = ((fake_samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)

            # FID
            fid.update(real_resized, real=True)
            fid.update(fake_resized, real=False)
            fid_val = fid.compute().item()

            # Inception Score
            is_mean, is_std = is_score(fake_resized)

            # KID
            kid.update(real_resized, real=True)
            kid.update(fake_resized, real=False)
            kid_mean, kid_std = kid.compute()

            metrics.append({
                "epoch": epoch+1,
                "loss_D": loss_D.item(),
                "loss_G": loss_G.item(),
                "FID": fid_val,
                "IS_mean": is_mean.item(),
                "IS_std": is_std.item(),
                "KID_mean": kid_mean.item(),
                "KID_std": kid_std.item()
            })

            pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    # Save generator
    torch.save(generator.state_dict(), os.path.join(out_dir, "generator.pth"))
    print("✅ Training complete. Model and metrics saved.")


# ------------------------
# Kaggle notebook usage (no argparse issues)
# ------------------------
if __name__ == "__main__":
    data_path = "/kaggle/input/dataset"  # <-- replace
    epochs = 50
    batch_size = 32  # can be small
    z_dim = 100
    img_size = 64
    lr = 0.0002
    out_dir = "/kaggle/working/outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    train(
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
        z_dim=z_dim,
        img_size=img_size,
        lr=lr,
        device=device,
        out_dir=out_dir
    )
