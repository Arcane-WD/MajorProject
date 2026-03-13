"""
Diagnostic test: Side-by-side comparison of Base CubiCasa Model vs Fine-Tuned Model.
Uses the EXACT pipeline methods as the main app.py.
"""
import torch
import cv2
import numpy as np
import pipeline
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

image_path = "sample io\\sample_image_4_highres.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"ERROR: Could not load image at {image_path}")
    exit(1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image loaded: {image.shape}")

# Ensure output directory exists
os.makedirs("sample io", exist_ok=True)

# ---- Models to compare ----
MODELS = {
    "cubicasa_base": "best_cleaner_model.pth",
    "finetuned_fpcad": "best_wall_model_finetuned.pth", 
}

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_path})")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [ERROR] {model_path} not found. Skipping.")
        continue

    model = pipeline.load_model_logic(model_path, device)

    # ==========================================
    # 1. FAST MODE (512px)
    # ==========================================
    print("\n--- Fast (512px) Inference ---")
    
    # 1a. Predict raw mask
    mask_fast = pipeline.predict_mask(model, device, image)
    print(f"  [Output] Mask size: {mask_fast.shape}, range: [{mask_fast.min():.3f}, {mask_fast.max():.3f}]")
    
    # Save the raw mask as PNG for verification
    cv2.imwrite(f"sample io/{model_name}_fast_mask.png", (mask_fast * 255).astype(np.uint8))
    
    # 1b. Process geometry (this includes Phase 5A vectorization + Phase 5B snapping/manhattan/gap close)
    vectors_fast = pipeline.process_geometry(mask_fast)
    print(f"  [Output] Plotted Vectors: {len(vectors_fast)}")
    
    # 1c. Generate 3D Mesh
    mesh_fast = pipeline.generate_3d_scene(vectors_fast)
    if mesh_fast:
        out_fast = f"sample io/{model_name}_fast_model.glb"
        mesh_fast.export(out_fast)
        print(f"  [Output] Exported GLB: {out_fast}")


    # ==========================================
    # 2. TILED MODE (Full Res)
    # ==========================================
    print("\n--- Tiled (High-Fidelity) Inference ---")
    
    # 2a. Predict tiled mask
    mask_tiled = pipeline.predict_tiled(model, device, image, progress_callback=lambda p: None)
    print(f"  [Output] Mask size: {mask_tiled.shape}, range: [{mask_tiled.min():.3f}, {mask_tiled.max():.3f}]")
    
    # Save the raw mask as PNG for verification
    cv2.imwrite(f"sample io/{model_name}_tiled_mask.png", (mask_tiled * 255).astype(np.uint8))
    
    # 2b. Process geometry
    vectors_tiled = pipeline.process_geometry(mask_tiled)
    print(f"  [Output] Plotted Vectors: {len(vectors_tiled)}")
    
    # 2c. Generate 3D Mesh
    mesh_tiled = pipeline.generate_3d_scene(vectors_tiled)
    if mesh_tiled:
        out_tiled = f"sample io/{model_name}_tiled_model.glb"
        mesh_tiled.export(out_tiled)
        print(f"  [Output] Exported GLB: {out_tiled}")

print(f"\n{'='*60}")
print("DONE! Compare the exported files in 'sample io/'")
