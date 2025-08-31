#!pip install torchaudio bindsnet librosa matplotlib
#run the above command in shell
import torch
import torch.nn as nn
from torchvision.utils import save_image

# ------------------------
# Generator class
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

# ------------------------
# Settings
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
generator_path = "/content/generator_final.pth"

output_image_path = "/content/generatedimage.jpg"
num_images = 1

# ------------------------
# Load generator and generate images
# ------------------------
generator = Generator(z_dim=z_dim).to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

noise = torch.randn(num_images, z_dim, 1, 1, device=device)
with torch.no_grad():
    fake_images = generator(noise)

save_image(fake_images, output_image_path, nrow=4, normalize=True)
print(f"✅ Generated {num_images} images saved to {output_image_path}")