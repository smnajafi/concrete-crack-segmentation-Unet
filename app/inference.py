import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.unet import UNet


MODEL_PATH = Path(__file__).parent.parent / "weights" / "best_model.pth"
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


class CrackModel:
    def __init__(self):
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, image: Image.Image):
        original_size = (image.height, image.width)
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.5).float()

        mask = mask.squeeze().cpu().numpy()

        # Resize mask back to original image dimensions
        mask_pil = Image.fromarray((mask * 255).astype("uint8"))
        mask_pil = mask_pil.resize((image.width, image.height), Image.NEAREST)
        mask = np.array(mask_pil) / 255.0

        return mask


def calculate_crack_score(mask: np.ndarray):
    crack_pixels = np.sum(mask)
    total_pixels = mask.size

    ratio = crack_pixels / total_pixels

    if ratio < 0.01:
        severity = "Low"
    elif ratio < 0.05:
        severity = "Medium"
    else:
        severity = "High"

    return ratio, severity