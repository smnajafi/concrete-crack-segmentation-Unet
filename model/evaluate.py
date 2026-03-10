import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CrackDataset
from unet import UNet


# -------------------------
# Config
# -------------------------
IMAGES_DIR = "data/raw/images"
MASKS_DIR = "data/raw/masks"
MODEL_PATH = "weights/best_model.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Transforms
# -------------------------
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float()),
])


# -------------------------
# Metrics
# -------------------------
def compute_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum((1, 2, 3))
    union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection

    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean().item()


def compute_dice(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum((1, 2, 3))
    total = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3))

    dice = (2 * intersection + 1e-8) / (total + 1e-8)
    return dice.mean().item()


# -------------------------
# Evaluation
# -------------------------
def evaluate():
    dataset = CrackDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_iou = 0
    total_dice = 0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_dice += compute_dice(outputs, masks)
            batches += 1

    print(f"Mean IoU: {total_iou / batches:.4f}")
    print(f"Mean Dice: {total_dice / batches:.4f}")


if __name__ == "__main__":
    evaluate()