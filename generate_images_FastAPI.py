import torch
import torch.nn as nn
from torchvision.utils import save_image
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid

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
# Setup
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
generator_path = "Replace with the path of your generator file"

generator = Generator(z_dim=z_dim).to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

# Create static folder
os.makedirs("outputs", exist_ok=True)

app = FastAPI()
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ------------------------
# Routes
# ------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>GAN Image Generator</title>
    </head>
    <body style="text-align:center; font-family: Arial;">
        <h1>GAN Image Generator</h1>
        <button onclick="generateImage()">Generate Image</button>
        <div id="result" style="margin-top:20px;"></div>

        <script>
        async function generateImage() {
            const response = await fetch('/generate');
            const data = await response.json();
            document.getElementById("result").innerHTML = `
                <h3>Generated Image:</h3>
                <img src="${data.url}" width="256"/><br><br>
                <a href="${data.url}" download="generated.png">
                    <button>Download</button>
                </a>
            `;
        }
        </script>
    </body>
    </html>
    """

@app.get("/generate")
def generate_image(num_images: int = 1):
    noise = torch.randn(num_images, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise)

    filename = f"outputs/generated_{uuid.uuid4().hex}.png"
    save_image(fake_images, filename, nrow=1, normalize=True)

    return {"url": f"/{filename}"}
