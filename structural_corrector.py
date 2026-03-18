"""
Phase 2C: Structural Corrector
Uses YOLOv8 door detections to retroactively correct wall gaps from Phase 5B.

Three correction scenarios:
  A) Fill false gaps — small wall breaks with no door bbox → close them
  B) Carve undetected doors — door bbox overlapping unbroken wall → split wall
  C) Resize oversized gaps — gap much larger than door bbox → shrink to fit
"""

import numpy as np
import math
from ultralytics import YOLO
import cv2

# --- PHASE 2C CONSTANTS (TUNABLE) ---
MAX_DOOR_WALL_DIST = 45.0
NOISE_GAP_THRESH = 25.0
COLLINEAR_TOL = 0.92
DOOR_CLASSES = {
    "single_door", "double_door", "sliding_door",
    "window", "bay_window", "blind_window", "opening_symbol"
}
IGNORED_CLASSES = {"bath", "stair", "escalator", "class_31", "railing", "wall", "parking"}

# ADD THIS LINE:
USEFUL_CLASSES = DOOR_CLASSES  # Only door/window types are structurally actionable

def load_yolo_model(weights_path="best.pt"):
    """Load the trained YOLOv8 model."""
    model = YOLO(weights_path)
    return model


def run_yolo_inference(model, image, conf=0.20):
    """
    Run YOLO inference with proper preprocessing (inversion for white backgrounds)
    Returns list of dicts: {class, x_center, y_center, width, height, x1, y1, x2, y2, confidence}
    """
    # 1. Preprocess: check brightness and invert if it's a white-background plan
    img_bgr = image.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness > 127:
        img_bgr = cv2.bitwise_not(img_bgr)
        print(f"  [2C-YOLO] Inverted white background (brightness {mean_brightness:.0f})")
    
    # 2. Run inference on the preprocessed image
    results = model.predict(img_bgr, conf=conf, verbose=False)
    detections = []

    if not results or len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detections.append({
            "class": cls_name,
            "confidence": conf_val,
            "x_center": (x1 + x2) / 2.0,
            "y_center": (y1 + y2) / 2.0,
            "width": x2 - x1,
            "height": y2 - y1,
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
        })

    return detections


def filter_detections(detections):
    """
    Step 1: Class filtering.
    Keep only structurally relevant classes and discard noise.
    """
    filtered = [d for d in detections if d["class"] not in IGNORED_CLASSES]
    print(f"  [2C-Filter] {len(detections)} raw → {len(filtered)} after class filter")
    return filtered


def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab, and the closest point on ab."""
    ap = p - a
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-10:
        return np.linalg.norm(ap), a
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest), closest


def _bbox_intersects_segment(det, p1, p2):
    """
    Check if a detection bounding box intersects or is very close to a wall segment.
    Uses the perpendicular distance from bbox center to the wall line.
    Returns (distance, closest_point_on_wall).
    """
    center = np.array([det["x_center"], det["y_center"]])
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    return _point_to_segment_dist(center, a, b)


def validate_doors_against_walls(detections, vectors):
    """
    Step 2: Door-Wall Consistency Check.
    Discard any door detection whose center is too far from any wall segment.
    """
    validated = []
    discarded = 0

    for det in detections:
        center = np.array([det["x_center"], det["y_center"]])
        best_dist = float('inf')

        for p1, p2 in vectors:
            a = np.array(p1, dtype=np.float64)
            b = np.array(p2, dtype=np.float64)
            dist, _ = _point_to_segment_dist(center, a, b)
            best_dist = min(best_dist, dist)

        if best_dist <= MAX_DOOR_WALL_DIST:
            det["wall_dist"] = best_dist
            det["status"] = "valid"
            validated.append(det)
        else:
            det["status"] = "discarded"
            discarded += 1

    print(f"  [2C-Validate] {len(detections)} detections → {len(validated)} valid, {discarded} discarded (too far from walls)")
    return detections  # Return ALL detections, now annotated with their status


def _are_collinear(p1, p2, p3, p4, angle_tol=COLLINEAR_TOL, dist_tol=5.0):
    """
    Check if two wall segments are geometrically collinear.
    This means they are parallel AND lie along the exact same infinite line.
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    
    if n1 < 1e-6 or n2 < 1e-6:
        return False
        
    # 1. Check parallel direction (cosine similarity)
    cos_sim = abs(np.dot(v1, v2) / (n1 * n2))
    if cos_sim < angle_tol:
        return False
        
    # 2. Check strict collinearity: distance from p3 and p4 to the infinite line (p1, p2)
    v1_normal = np.array([-v1[1], v1[0]]) / n1
    
    dist_p3 = abs(np.dot(np.array(p3) - np.array(p1), v1_normal))
    dist_p4 = abs(np.dot(np.array(p4) - np.array(p1), v1_normal))
    
    # Both endpoints of the second segment must be extremely close to the line
    return dist_p3 <= dist_tol and dist_p4 <= dist_tol


def _find_collinear_gap_pairs(vectors):
    """
    Find pairs of wall segments that are collinear and have a gap between them.
    Returns list of (idx_i, idx_j, gap_dist, gap_p1, gap_p2).
    """
    gap_pairs = []
    n = len(vectors)

    for i in range(n):
        p1_i, p2_i = vectors[i]
        for j in range(i + 1, n):
            p1_j, p2_j = vectors[j]

            if not _are_collinear(p1_i, p2_i, p1_j, p2_j):
                continue

            # Find the closest endpoint pair between the two walls
            endpoints_i = [np.array(p1_i), np.array(p2_i)]
            endpoints_j = [np.array(p1_j), np.array(p2_j)]

            best_dist = float('inf')
            best_pair = None

            for ei_idx, ei in enumerate(endpoints_i):
                for ej_idx, ej in enumerate(endpoints_j):
                    d = np.linalg.norm(ei - ej)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ei_idx, ej_idx, ei, ej)

            if best_pair and best_dist > 1.0:  # There IS a gap (not already connected)
                gap_pairs.append((i, j, best_dist, best_pair[2], best_pair[3], best_pair[0], best_pair[1]))

    return gap_pairs


def normalize_wall_gaps(vectors, door_detections):
    """
    Step 3: Wall Gap Normalization — the core algorithm.

    Scenario A: Fill gaps that have no door bbox nearby.
    Scenario B: Carve gaps where a door bbox overlaps an unbroken wall.
    Scenario C: Resize gaps that are much larger than the door bbox.
    """
    # Separate door-type detections for gap logic
    door_dets = [d for d in door_detections if d["class"] in DOOR_CLASSES]
    print(f"  [2C-Normalize] Working with {len(door_dets)} door-type detections and {len(vectors)} wall segments")

    # --- Scenario A: Fill false gaps ---
    gap_pairs = _find_collinear_gap_pairs(vectors)
    gaps_filled = 0
    gaps_to_merge = set()  # (i, j) pairs to merge

    for (i, j, gap_dist, gap_p1, gap_p2, ei_idx, ej_idx) in gap_pairs:
        if gap_dist > NOISE_GAP_THRESH:
            continue  # Gap is too large to be noise

        # Check if any door bbox center falls within this gap region
        gap_center = (gap_p1 + gap_p2) / 2.0
        gap_has_door = False

        for det in door_dets:
            det_center = np.array([det["x_center"], det["y_center"]])
            dist_to_gap = np.linalg.norm(det_center - gap_center)
            # If a door is within half the gap distance, it's a real door gap
            if dist_to_gap < gap_dist * 0.75:
                gap_has_door = True
                break

        if not gap_has_door:
            gaps_to_merge.add((i, j, ei_idx, ej_idx))
            gaps_filled += 1

    # Apply gap fills by extending endpoints to meet
    modified_vectors = list(vectors)
    merged_indices = set()

    for (i, j, ei_idx, ej_idx) in gaps_to_merge:
        p1_i, p2_i = modified_vectors[i]
        p1_j, p2_j = modified_vectors[j]

        # Compute midpoint of the gap
        ep_i = np.array(p2_i if ei_idx == 1 else p1_i)
        ep_j = np.array(p2_j if ej_idx == 1 else p1_j)
        midpoint = tuple((ep_i + ep_j) / 2.0)

        # Extend both walls to meet at the midpoint
        if ei_idx == 0:
            modified_vectors[i] = (midpoint, p2_i)
        else:
            modified_vectors[i] = (p1_i, midpoint)

        if ej_idx == 0:
            modified_vectors[j] = (midpoint, p2_j)
        else:
            modified_vectors[j] = (p1_j, midpoint)
            
        print(f"    [Scenario A] Filled false gap of {gap_dist:.1f}px (Limit: {NOISE_GAP_THRESH}px)")

    print(f"  [2C-ScenarioA] Filled {gaps_filled} false gaps (no door present)")

    # --- Scenario B: Carve undetected door openings ---
    doors_carved = 0
    new_vectors = []
    walls_to_skip = set()

    for det in door_dets:
        det_center = np.array([det["x_center"], det["y_center"]])

        for vi in range(len(modified_vectors)):
            if vi in walls_to_skip:
                continue

            p1 = np.array(modified_vectors[vi][0], dtype=np.float64)
            p2 = np.array(modified_vectors[vi][1], dtype=np.float64)

            dist, closest = _point_to_segment_dist(det_center, p1, p2)

            if dist > MAX_DOOR_WALL_DIST:
                continue

            # Check if the door bbox is fully INSIDE this wall segment (wall covers the door)
            wall_vec = p2 - p1
            wall_len = np.linalg.norm(wall_vec)
            if wall_len < 1.0:
                continue

            wall_dir = wall_vec / wall_len

            # Project door bbox edges onto the wall direction
            # Determine door width along the wall
            bbox_corners = [
                np.array([det["x1"], det["y1"]]),
                np.array([det["x2"], det["y1"]]),
                np.array([det["x1"], det["y2"]]),
                np.array([det["x2"], det["y2"]]),
            ]
            projections = [np.dot(c - p1, wall_dir) for c in bbox_corners]
            proj_min = max(0, min(projections))
            proj_max = min(wall_len, max(projections))

            door_span = proj_max - proj_min

            if door_span < 5.0:  # Door bbox doesn't meaningfully overlap this wall
                continue

            # Check if there's already a gap here (gap pairs cover this)
            # If wall is continuous through the door area, we need to carve
            # Split: wall[p1 → cut_start] + gap + wall[cut_end → p2]
            cut_start_pt = p1 + proj_min * wall_dir
            cut_end_pt = p1 + proj_max * wall_dir

            # Only carve if the door is substantially inside the wall
            if proj_min > 2.0 and (wall_len - proj_max) > 2.0:
                walls_to_skip.add(vi)
                new_vectors.append((tuple(p1), tuple(cut_start_pt)))
                new_vectors.append((tuple(cut_end_pt), tuple(p2)))
                doors_carved += 1
                break  # This door has been handled

    # Rebuild the vector list
    final_vectors = []
    for vi in range(len(modified_vectors)):
        if vi not in walls_to_skip:
            final_vectors.append(modified_vectors[vi])
    final_vectors.extend(new_vectors)

    print(f"  [2C-ScenarioB] Carved {doors_carved} new door openings in continuous walls")

    # --- Scenario C: Resize oversized gaps ---
    # Re-scan for collinear gaps that are oversized relative to door bbox
    gap_pairs_post = _find_collinear_gap_pairs(final_vectors)
    gaps_resized = 0

    for (i, j, gap_dist, gap_p1, gap_p2, ei_idx, ej_idx) in gap_pairs_post:
        gap_center = (gap_p1 + gap_p2) / 2.0
        gap_dir = gap_p2 - gap_p1
        gap_dir_norm = gap_dir / np.linalg.norm(gap_dir)

        for det in door_dets:
            det_center = np.array([det["x_center"], det["y_center"]])
            
            # 1. Door center must be strictly near the infinite line connecting the gap endpoints
            v1_normal = np.array([-gap_dir_norm[1], gap_dir_norm[0]])
            dist_to_line = abs(np.dot(det_center - gap_p1, v1_normal))
            if dist_to_line > MAX_DOOR_WALL_DIST:
                continue

            # 2. Door center must physically overlap the gap (close to gap center)
            dist_to_gap_center = np.linalg.norm(det_center - gap_center)

            # 3. Gap cannot be ridiculously large compared to the door length
            door_width_along_gap = abs(det["width"] * gap_dir_norm[0]) + abs(det["height"] * gap_dir_norm[1])
            
            if dist_to_gap_center > door_width_along_gap * 1.0:
                continue # Door is not centered in the gap

            if door_width_along_gap < 5.0:
                continue

            if gap_dist > door_width_along_gap * 1.5 and gap_dist <= door_width_along_gap * 3.0:
                print(f"    [Scenario C] Resizing gap {gap_dist:.1f}px to match door {door_width_along_gap:.1f}px")
                # Shrink gap to match door width
                shrink_amount = (gap_dist - door_width_along_gap) / 2.0
                p1_i, p2_i = final_vectors[i]
                p1_j, p2_j = final_vectors[j]

                # ADD `shrink_amount * gap_dir_norm` to move towards gap_p2
                if ei_idx == 1:
                    new_ep = np.array(p2_i) + shrink_amount * gap_dir_norm
                    final_vectors[i] = (p1_i, tuple(new_ep))
                else:
                    new_ep = np.array(p1_i) + shrink_amount * gap_dir_norm
                    final_vectors[i] = (tuple(new_ep), p2_i)

                # SUBTRACT `shrink_amount * gap_dir_norm` to move towards gap_p1
                if ej_idx == 1:
                    new_ep = np.array(p2_j) - shrink_amount * gap_dir_norm
                    final_vectors[j] = (p1_j, tuple(new_ep))
                else:
                    new_ep = np.array(p1_j) - shrink_amount * gap_dir_norm
                    final_vectors[j] = (tuple(new_ep), p2_j)

                gaps_resized += 1
                break  # One door per gap
            else:
                print(f"    [Scenario C] Skipped gap {gap_dist:.1f}px vs door {door_width_along_gap:.1f}px (ratio {gap_dist/door_width_along_gap:.1f}x)")

    print(f"  [2C-ScenarioC] Resized {gaps_resized} oversized gaps to match door width")
    print(f"  [2C-Result] {len(vectors)} input vectors → {len(final_vectors)} output vectors")

    return final_vectors


def correct_structure(image, vectors, weights_path="best.pt"):
    """
    Main entry point for Phase 2C.
    Takes the original image and Phase 5B vectors, returns corrected vectors.
    """
    print("\n=== Phase 2C: Structural Correction ===")

    # 1. Load YOLO and run inference
    model = load_yolo_model(weights_path)
    detections = run_yolo_inference(model, image)
    print(f"  [2C] YOLO detected {len(detections)} objects")

    if not detections:
        print("  [2C] No detections — returning vectors unchanged")
        return vectors, []

    filtered_dets = []
    ignored = 0
    for det in detections:
        if det["class"] in USEFUL_CLASSES:
            filtered_dets.append(det)
        else:
            det["status"] = "ignored"
            ignored += 1
            
    print(f"  [2C-Filter] {len(detections)} raw → {len(filtered_dets)} after class filter")
    
    # 3. Validate against walls
    # (validate_doors_against_walls now annotates with 'valid' or 'discarded')
    validate_doors_against_walls(filtered_dets, vectors)
    
    # Also keep the ignored ones in the final output for completeness
    for det in detections:
        if "status" not in det:
            det["status"] = "ignored"

    # 4. Normalize Wall Gaps
    # Only use valid doors for structural correction
    door_dets = [d for d in detections if d.get("status") == "valid" and d["class"] in DOOR_CLASSES]
    
    # Tag them specifically so the UI knows these were the ones used for walls
    for d in door_dets:
        d["status"] = "valid_door"
        
    final_vectors = normalize_wall_gaps(vectors, door_dets)

    return final_vectors, detections


if __name__ == "__main__":
    import cv2
    # Quick test
    img = cv2.imread("sampleio/sample_image_4_highres.jpg")
    if img is None:
        print("ERROR: Could not load test image")
        exit(1)

    # We need vectors from the pipeline to test
    import pipeline
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipeline.load_model_logic("best_cleaner_model_v3.pth", device)
    mask = pipeline.predict_tiled(model, device, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    vectors,_ = pipeline.process_geometry(mask)

    corrected, dets = correct_structure(img, vectors)
    print(f"\nOriginal: {len(vectors)} vectors → Corrected: {len(corrected)} vectors")
    print(f"Detections used: {len(dets)}")
