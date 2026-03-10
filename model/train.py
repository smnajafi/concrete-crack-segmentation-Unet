from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CrackDataset
from unet import UNet
from loss import DiceBCELoss


# -------------------------
# Config
# -------------------------
IMAGES_DIR = "data/raw/images"
MASKS_DIR = "data/raw/masks"
WEIGHTS_DIR = "weights"
MODEL_NAME = "best_model.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
RANDOM_SEED = 42

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


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
# Dataset and loaders
# -------------------------
def get_dataloaders():
    # Split on base samples first, then apply augmentation only to train
    base_dataset = CrackDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
    )

    total_size = len(base_dataset)
    train_size = int(total_size * TRAIN_SPLIT)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_image_paths = [base_dataset.samples[i][0] for i in train_indices]
    train_mask_paths = [base_dataset.samples[i][1] for i in train_indices]
    val_image_paths = [base_dataset.samples[i][0] for i in val_indices]
    val_mask_paths = [base_dataset.samples[i][1] for i in val_indices]

    train_dataset = CrackDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=True,
    )
    # Override samples with train split only
    train_dataset.samples = list(zip(train_image_paths, train_mask_paths))
    train_dataset.augmented_samples = [
        (img, msk, angle)
        for img, msk in train_dataset.samples
        for angle in [0, 90, 180, 270]
    ]

    val_dataset = CrackDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        image_transform=image_transform,
        mask_transform=mask_transform,
        augment=False,
    )
    val_dataset.samples = list(zip(val_image_paths, val_mask_paths))
    val_dataset.augmented_samples = [(img, msk, 0) for img, msk in val_dataset.samples]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# -------------------------
# Training and validation
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(loader)


# -------------------------
# Main
# -------------------------
def main():
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders()

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_path = Path(WEIGHTS_DIR) / MODEL_NAME

    log_path = Path(WEIGHTS_DIR) / "training_log.txt"
    print(f"Using device: {DEVICE}")
    print(f"Real-time log: {log_path}")

    with open(log_path, "w") as log_file:
        log_file.write(f"Training started | device={DEVICE} | epochs={NUM_EPOCHS}\n")
        log_file.flush()

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)

            line = (
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            print(line)
            log_file.write(line + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                saved_line = f"  -> Saved best model (val_loss={val_loss:.4f})"
                print(saved_line)
                log_file.write(saved_line + "\n")

            log_file.flush()

        log_file.write("Training finished.\n")
    print("Training finished.")


if __name__ == "__main__":
    main()