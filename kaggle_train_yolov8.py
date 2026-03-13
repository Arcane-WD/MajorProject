"""
Kaggle Training Script for Phase 2B: YOLOv8 Icon Detection
Dataset: https://www.kaggle.com/datasets/catwhisker/floorplancad-dataset

Instructions for Kaggle:
1. Create a New Notebook.
2. Set Accelerator to GPU P100 (or T4 x2).
3. Click '+ Add Data' and search for 'catwhisker floorplancad'. Add it.
4. Add a cell and run: `!pip install ultralytics`
5. Paste this script into the next cell and run it!
6. When done, download `/kaggle/working/runs/detect/floorplan_icons/weights/best.pt`
"""

import os
from ultralytics import YOLO

def train_yolo():
    print("Initializing YOLOv8 Medium...")
    # Load the official pre-trained medium model (good balance of speed/accuracy)
    model = YOLO("yolov8m.pt")

    # The dataset has a data.yaml that tells YOLO where the images/labels are
    # and what the 28 class names are.
    yaml_path = "/kaggle/input/floorplancad-dataset/data.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: Could not find {yaml_path}. Did you add the dataset to the notebook?")
        return
        
    print(f"✅ Found dataset config at {yaml_path}")
    print("🚀 Starting Training on GPU...")

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=60,            # 60 epochs is usually enough for fine-tuning
        imgsz=640,            # Dataset images are 640x640
        batch=16,             # Fits well in a 16GB Kaggle GPU
        device=0,             # Use the primary GPU
        augment=True,         # Auto-augmentation to prevent overfitting
        mosaic=1.0,           # Mixes 4 images together to train better context
        name="floorplan_icons",
        patience=15,          # Stop early if no improvement for 15 epochs
        save=True             # Save the best weights
    )

    print("🎉 Training Complete!")
    print("Evaluating Best Model on Validation Set...")
    
    # Validate the best model
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.3f}")
    
    print("\n✅ Done! Download your model weights from:")
    print("   /kaggle/working/runs/detect/floorplan_icons/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
