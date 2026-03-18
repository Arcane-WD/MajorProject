import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --------------------------
# 1. Define the Fast Dataset
# --------------------------
class FastFloorplanDataset(Dataset):
    def __init__(self, root):
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "masks")
        # Sort filenames to ensure images match masks
        self.filenames = sorted(os.listdir(self.images_dir))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # ImageNet stats matching the Pre-trained ResNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        # Load Image
        img_path = os.path.join(self.images_dir, self.filenames[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load Mask (Grayscale)
        mask_path = os.path.join(self.masks_dir, self.filenames[i])
        mask = cv2.imread(mask_path, 0)

        # Apply Transforms
        img = self.transform(img)
        
        # Normalize Mask to 0.0 - 1.0
        mask = torch.from_numpy(mask).float() / 255.0
        mask = mask.unsqueeze(0) # [H, W] -> [1, H, W]

        return img, mask

# --------------------------
# 2. Setup Training Config
# --------------------------
# Ensure this matches the folder your preprocessing script created
PROCESSED_DIR = '/kaggle/input/processed-dataset-512/processed_dataset_512' # Update this path if needed

# Hyperparameters
BATCH_SIZE = 8  # Safe for Colab T4 GPU / Kaggle
EPOCHS = 30     # Resume training for more epochs
LR = 0.0001     # Learning Rate

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load Data
train_dataset = FastFloorplanDataset(PROCESSED_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"Loaded {len(train_dataset)} images for training.")

# --- MODEL & OPTIMIZER ---
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=1
).to(device)

# Load existing weights to resume training
WEIGHTS_PATH = "/kaggle/input/your-weights-dataset/best_cleaner_model.pth" # Update this path
if os.path.exists(WEIGHTS_PATH):
    print(f"Resuming training from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
else:
    print(f"Weights not found at {WEIGHTS_PATH}. Starting from scratch.")

optimizer = Adam(model.parameters(), lr=LR)
criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
scaler = GradScaler() # Initializes Mixed Precision Scaler

# --- OPTIMIZED TRAINING LOOP ---
best_loss = float("inf")

print(f"Starting Training for {EPOCHS} epochs on {device} (Mixed Precision ON)...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, masks in progress_bar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_cleaner_model.pth")
        print(f"--> Model Saved (Loss: {best_loss:.4f})")

print("Training Complete.")
