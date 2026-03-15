"""
Kaggle Training Script for Phase 2B: YOLOv8 Icon Detection
Using FiftyOne to instantly pull Voxel51/FloorPlanCAD from HuggingFace, convert it to YOLO, and start training!

Instructions for Kaggle:
1. Create a New Notebook.
2. Set Accelerator to GPU P100 (or T4 x2) and turn ON Internet access under Notebook Options.
3. Don't worry about adding any datasets manually. The script handles it.
4. Add a cell and run: `!pip install fiftyone ultralytics huggingface_hub[hf_transfer]`
5. Paste this script into the next cell and run it!
6. When done, download `/kaggle/working/runs/detect/floorplan_icons/weights/best.pt`
"""

import os
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from ultralytics import YOLO

def prepare_yolo_dataset(export_dir="/kaggle/working/yolo_dataset"):
    print("📥 Downloading FloorPlanCAD via FiftyOne (HuggingFace)...")
    print("This might take a few minutes as it pulls the pristine dataset block.")
    
    # Load dataset exactly as the datacard requested
    try:
        dataset = load_from_hub("Voxel51/FloorPlanCAD")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(f"✅ Successfully loaded {len(dataset)} images from HuggingFace.")
    
    # FloorPlanCAD on HF might not have native train/val splits predefined in the Hub args.
    # We will randomly split it 90% train / 10% val for YOLO.
    dataset.shuffle()
    train_count = int(len(dataset) * 0.9)
    train_view = dataset[:train_count]
    val_view = dataset[train_count:]

    print("📤 Exporting to YOLO format...")
    
    # FloorPlanCAD has rare classes. The val split might miss some. 
    # We must explicitly compute and pass the global class list so FiftyOne doesn't crash on mismatch.
    try:
        classes = dataset.default_classes if dataset.default_classes else dataset.distinct("ground_truth.detections.label")
    except Exception:
        # Hardcoded fallback exactly from dataset docs just in case
        classes = ['single_door', 'double_door', 'sliding_door', 'window', 'bay_window', 'blind_window', 
                   'opening_symbol', 'stair', 'gas_stove', 'refrigerator', 'washing_machine', 'sofa', 
                   'bed', 'chair', 'table', 'bedside_cupboard', 'tv_cabinet', 'half_height_cabinet', 
                   'high_cabinet', 'wardrobe', 'sink', 'bath', 'bath_tub', 'squat_toilet', 'urinal', 
                   'toilet', 'elevator', 'escalator', 'wall', 'parking']

    print(f"📋 Enforcing {len(classes)} global classes to prevent split mismatch...")
    
    # Export train split
    train_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=classes
    )
    
    # Export val split
    val_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val",
        classes=classes
    )
    
    yaml_path = os.path.join(export_dir, "dataset.yaml")
    return yaml_path

def train_yolo():
    export_dir = "/kaggle/working/yolo_dataset"
    yaml_path = os.path.join(export_dir, "dataset.yaml")
    
    print("\n==============================================")
    print("🚀 YOLOv8 FloorPlanCAD Training Initializing...")
    print("==============================================\n")
    
    # If the dataset hasn't been pulled and converted yet, do it now.
    if not os.path.exists(yaml_path):
        yaml_path = prepare_yolo_dataset(export_dir)
        if not yaml_path:
            return

    print(f"✅ Dataset prepared and converted to YOLO format at {yaml_path}!")
    print("🚀 Initializing YOLOv8 Medium...")
    
    # Load the official pre-trained medium model
    model = YOLO("yolov8m.pt")

    print(f"✅ Starting Training on GPU...")

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=50,            # 50 epochs
        imgsz=640,            # Target size 640x640
        batch=16,             # Fits well in a 16GB Kaggle GPU
        device=0,             # Use the primary GPU
        augment=True,         # Auto-augmentation
        mosaic=1.0,           # Mixes 4 images together to train better context
        name="floorplan_icons",
        patience=10,          # Stop early if no improvement for 10 epochs
        save=True             # Save the best weights
    )

    print("🎉 Training Complete!")
    print("Evaluating Best Model on Validation Set...")
    
    # Validate the best model
    try:
        metrics = model.val()
        print(f"Validation mAP50: {metrics.box.map50:.3f}")
    except Exception as dict_e:
        print(f"Validation finished. Metrics saved to runs/detect/floorplan_icons.")
    
    print("\n✅ Done! Download your model weights from:")
    print("   /kaggle/working/runs/detect/floorplan_icons/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
