"""
Direct Pipeline (Production Mode)
──────────────────────────────────
Headless, uninterrupted execution.
Input: floor plan image → Output: GLB files (pre-correction + post-correction)

Usage:
    python main_direct.py                           # uses defaults
    python main_direct.py --image path/to/img.jpg   # custom image
    python main_direct.py --no-yolo                  # skip Phase 2C
"""

import argparse
import os
import sys
import time
import cv2
import torch
import numpy as np

import pipeline
import structural_corrector


def ensure_output_dirs():
    """Create the tiered output directory structure."""
    dirs = [
        "outputs/direct_results/GLBs",
        "outputs/direct_results/final_render_icons",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0]  # GLBs path


def run_direct(image_path, yolo_weights="best.pt", skip_yolo=False):
    """
    Full headless pipeline: Image → Mask → Vectors → [Structural Correction] → GLB
    Exports BOTH pre-correction and post-correction GLBs for visual comparison.
    """
    print("=" * 60)
    print("  DIRECT PIPELINE — Production Mode")
    print("=" * 60)

    glb_dir = ensure_output_dirs()
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image at {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  [Input] {image_path} ({image.shape[1]}x{image.shape[0]})")

    # 2. Load U-Net model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipeline.load_model_logic("best_cleaner_model_v3.pth", device)
    print(f"  [Model] U-Net loaded on {device}")

    # 3. Predict wall mask (always use tiled for production quality)
    t0 = time.time()
    mask = pipeline.predict_tiled(model, device, image_rgb, progress_callback=lambda p: None)
    print(f"  [Mask] Predicted in {time.time()-t0:.1f}s, shape={mask.shape}")

    # 4. Geometry extraction (Phase 5A + 5B, NO structural correction yet)
    vectors_raw, _ = pipeline.process_geometry(mask)
    print(f"  [Geometry] {len(vectors_raw)} vectors after Phase 5B")

    # 5. Export PRE-correction GLB
    mesh_pre = pipeline.generate_3d_scene(vectors_raw)
    if mesh_pre:
        pre_path = os.path.join(glb_dir, f"{basename}_pre_correction.glb")
        mesh_pre.export(pre_path)
        print(f"  [Export] PRE-correction GLB → {pre_path}")

    # 6. Structural Correction (Phase 2C)
    if not skip_yolo and os.path.exists(yolo_weights):
        vectors_corrected, detections = structural_corrector.correct_structure(
            image, vectors_raw, weights_path=yolo_weights
        )

        # Export POST-correction GLB
        mesh_post = pipeline.generate_3d_scene(vectors_corrected)
        if mesh_post:
            post_path = os.path.join(glb_dir, f"{basename}_post_correction.glb")
            mesh_post.export(post_path)
            print(f"  [Export] POST-correction GLB → {post_path}")

        print(f"\n  COMPARISON: {len(vectors_raw)} vectors (pre) → {len(vectors_corrected)} vectors (post)")
        print(f"  Detections used: {len(detections)}")
    else:
        reason = "skipped by flag" if skip_yolo else f"{yolo_weights} not found"
        print(f"  [2C] Structural Correction {reason} — only pre-correction GLB exported")

    print("\n" + "=" * 60)
    print(f"  DONE — Output in: {glb_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Pipeline — Scan-to-BIM")
    parser.add_argument("--image", default="sampleio/sample_image_4_highres.jpg",
                        help="Path to the floor plan image")
    parser.add_argument("--yolo-weights", default="best.pt",
                        help="Path to YOLOv8 weights")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip Phase 2C structural correction")
    args = parser.parse_args()

    run_direct(args.image, args.yolo_weights, args.no_yolo)
