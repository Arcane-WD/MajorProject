"""
Demonstration Pipeline (Step-by-Step Mode)
──────────────────────────────────────────
Saves every intermediate result as a numbered artifact for presentation.
Each run gets its own timestamped directory.

Usage:
    python main_demons.py                           # uses defaults
    python main_demons.py --image path/to/img.jpg   # custom image
    python main_demons.py --no-yolo                  # skip Phase 2C

Output structure:
    outputs/demonstration_results/{timestamp}/
        ├── 00_input_image.jpg
        ├── 01_raw_mask.png
        ├── 02_refined_mask.png
        ├── 03_skeleton.png
        ├── 04_vectors_pre5b.png
        ├── 05_vectors_post5b.png
        ├── 06_yolo_detections.png
        ├── 07_vectors_post2c.png
        ├── 08_model_pre_correction.glb
        └── 09_model_post_correction.glb
"""

import argparse
import os
import sys
import time
import cv2
import torch
import numpy as np
from datetime import datetime
from skimage.morphology import skeletonize

import pipeline
import structural_corrector


def create_run_dir():
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", "demonstration_results", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def draw_vectors_on_image(image_shape, vectors, color=(0, 255, 0), thickness=2):
    """Draw wall vectors as colored lines on a blank canvas."""
    canvas = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    for p1, p2 in vectors:
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(canvas, pt1, pt2, color, thickness)
    return canvas

def draw_doors_on_vector_map(canvas, detections):
    """Draw thick colored regions for detected valid doors over the vector map."""
    canvas_out = canvas.copy()
    for det in detections:
        if det.get("status") == "valid_door":
            x_c, y_c = int(det["x_center"]), int(det["y_center"])
            w, h = int(det["width"]), int(det["height"])
            cv2.rectangle(canvas_out, (x_c - w//2, y_c - h//2), (x_c + w//2, y_c + h//2), (0, 165, 255), -1) # Orange filled box
    return canvas_out

def draw_yolo_detections(image, detections):
    """Draw YOLO bounding boxes with labels on a copy of the image."""
    overlay = image.copy()
    colors = {
        "single_door": (0, 255, 0),
        "double_door": (255, 255, 0),
        "sliding_door": (0, 255, 255),
        "window": (255, 0, 0),
        "bay_window": (255, 128, 0),
        "blind_window": (128, 0, 255),
    }
    default_color = (200, 200, 200)

    for det in detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        cls = det["class"]
        conf = det["confidence"]
        color = colors.get(cls, default_color)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(overlay, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return overlay


def run_demonstration(image_path, yolo_weights="best.pt", skip_yolo=False):
    """
    Step-by-step pipeline with all intermediates saved as numbered files.
    """
    print("=" * 60)
    print("  DEMONSTRATION PIPELINE — Step-by-Step Mode")
    print("=" * 60)

    run_dir = create_run_dir()
    step = 0

    # ── Step 0: Save input image ──
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image at {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_path = os.path.join(run_dir, f"{step:02d}_input_image.jpg")
    cv2.imwrite(save_path, image)
    print(f"  [{step:02d}] Input image saved → {save_path}")
    step += 1

    # ── Step 1: Load model and predict raw mask ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipeline.load_model_logic("best_cleaner_model_v3.pth", device)

    t0 = time.time()
    mask = pipeline.predict_tiled(model, device, image_rgb, progress_callback=lambda p: None)
    elapsed = time.time() - t0

    mask_vis = (mask * 255).astype(np.uint8)
    save_path = os.path.join(run_dir, f"{step:02d}_raw_mask.png")
    cv2.imwrite(save_path, mask_vis)
    print(f"  [{step:02d}] Raw probability mask saved ({elapsed:.1f}s) → {save_path}")
    step += 1

    # ── Step 2: Refine mask ──
    clean_mask = pipeline.refine_mask(mask)
    save_path = os.path.join(run_dir, f"{step:02d}_refined_mask.png")
    cv2.imwrite(save_path, clean_mask)
    print(f"  [{step:02d}] Refined binary mask saved → {save_path}")
    step += 1

    # ── Step 3: Skeleton ──
    skeleton = skeletonize(clean_mask > 0).astype(np.uint8) * 255
    save_path = os.path.join(run_dir, f"{step:02d}_skeleton.png")
    cv2.imwrite(save_path, skeleton)
    print(f"  [{step:02d}] Skeleton topology saved → {save_path}")
    step += 1

    # ── Step 4: Raw vectors (pre-5B) ──
    # Rebuild graph manually for demonstration
    import networkx as nx
    skel_binary = skeleton // 255
    y, x = np.where(skel_binary > 0)
    points = list(zip(x.tolist(), y.tolist()))
    points_set = set(points)
    G = nx.Graph()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for u, v in points:
        G.add_node((u, v))
        for du, dv in shifts:
            neighbor = (u + du, v + dv)
            if neighbor in points_set and (u, v) < neighbor:
                G.add_edge((u, v), neighbor, weight=1.0)

    vectors_raw = pipeline.vectorize_hybrid(G, clean_mask)
    # Prune short/duplicate
    final_raw = []
    seen = set()
    for p1, p2 in vectors_raw:
        t_p1 = tuple(np.round(p1).astype(int))
        t_p2 = tuple(np.round(p2).astype(int))
        edge = tuple(sorted((t_p1, t_p2)))
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        if edge not in seen and length > pipeline.MIN_LENGTH:
            seen.add(edge)
            final_raw.append((p1, p2))

    canvas_pre5b = draw_vectors_on_image(image.shape, final_raw, color=(0, 200, 255))
    save_path = os.path.join(run_dir, f"{step:02d}_vectors_pre5b.png")
    cv2.imwrite(save_path, canvas_pre5b)
    print(f"  [{step:02d}] Pre-5B vectors ({len(final_raw)} segments) saved → {save_path}")
    step += 1

    # ── Step 5: Post-5B vectors ──
    vectors_5b = pipeline.snap_vertices(final_raw)
    vectors_5b = pipeline.enforce_manhattan(vectors_5b)
    vectors_5b = pipeline.close_gaps(vectors_5b)

    canvas_post5b = draw_vectors_on_image(image.shape, vectors_5b, color=(0, 255, 0))
    save_path = os.path.join(run_dir, f"{step:02d}_vectors_post5b.png")
    cv2.imwrite(save_path, canvas_post5b)
    print(f"  [{step:02d}] Post-5B vectors ({len(vectors_5b)} segments) saved → {save_path}")
    step += 1

    # ── Step 6: YOLO detections ──
    vectors_final = vectors_5b
    detections = []
    all_tracked_dets = []

    if not skip_yolo and os.path.exists(yolo_weights):
        vectors_corrected, all_tracked_dets, door_cuts = structural_corrector.correct_structure(image, vectors_5b, weights_path=yolo_weights)
        vectors_final = vectors_corrected
        
        discarded_doors = [d for d in all_tracked_dets if d.get("status") == "discarded" and d["class"] in structural_corrector.DOOR_CLASSES]
        valid_doors = [d for d in all_tracked_dets if d.get("status") == "valid_door"]

        yolo_vis_valid = draw_yolo_detections(image, valid_doors)
        save_path_valid = os.path.join(run_dir, f"{step:02d}a_yolo_used_doors.png")
        cv2.imwrite(save_path_valid, yolo_vis_valid)

        yolo_vis_discard = draw_yolo_detections(image, discarded_doors)
        save_path_discard = os.path.join(run_dir, f"{step:02d}b_yolo_discarded_doors.png")
        cv2.imwrite(save_path_discard, yolo_vis_discard)
        print(f"  [{step:02d}] YOLO doors ({len(valid_doors)} used, {len(discarded_doors)} discarded) saved")
        step += 1

        # ── Step 7: Post-2C vectors ──
        canvas_post2c = draw_vectors_on_image(image.shape, vectors_corrected, color=(0, 255, 128))
        canvas_post2c = draw_doors_on_vector_map(canvas_post2c, all_tracked_dets)
        save_path = os.path.join(run_dir, f"{step:02d}_vectors_post2c.png")
        cv2.imwrite(save_path, canvas_post2c)
        print(f"  [{step:02d}] Post-2C corrected vectors ({len(vectors_corrected)} segments) saved → {save_path}")
        step += 1
    else:
        reason = "skipped by flag" if skip_yolo else f"{yolo_weights} not found"
        print(f"  [{step:02d}] YOLO {reason} — skipping Phase 2C")
        step += 1

    # ── Step 8: Pre-correction GLB ──
    mesh_pre, _ = pipeline.generate_3d_scene(vectors_5b)
    if mesh_pre:
        save_path = os.path.join(run_dir, f"{step:02d}_model_pre_correction.glb")
        mesh_pre.export(save_path)
        print(f"  [{step:02d}] Pre-correction 3D model saved → {save_path}")
    step += 1

    # ── Step 9: Post-correction GLB ──
    mesh_post, _ = pipeline.generate_3d_scene(vectors_final, all_tracked_dets, door_cuts if 'door_cuts' in locals() else None)
    if mesh_post:
        save_path = os.path.join(run_dir, f"{step:02d}_model_post_correction.glb")
        mesh_post.export(save_path)
        print(f"  [{step:02d}] Post-correction 3D model saved → {save_path}")
    step += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  DEMONSTRATION COMPLETE")
    print(f"  Total steps: {step}")
    print(f"  Output directory: {run_dir}")
    print(f"  Vectors: {len(final_raw)} raw → {len(vectors_5b)} post-5B → {len(vectors_final)} final")
    if detections:
        print(f"  YOLO detections used: {len(detections)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration Pipeline — Scan-to-BIM")
    parser.add_argument("--image", default="sampleio/sample_image_4_highres.jpg",
                        help="Path to the floor plan image")
    parser.add_argument("--yolo-weights", default="best.pt",
                        help="Path to YOLOv8 weights")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip Phase 2C structural correction")
    args = parser.parse_args()

    run_demonstration(args.image, args.yolo_weights, args.no_yolo)
