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
    "single_door", "double_door", "sliding_door"
}
IGNORED_CLASSES = {"stair", "escalator", "class_31", "railing", "wall", "parking","bay_window","blind_window", "opening_symbol","squat_toilet", "bath","sink"}
USEFUL_CLASSES = {"single_door", "double_door", "sliding_door", "window", "bed", "half_height_cabinet", "kitchen_cabinet", "sofa", "table", "chair", }

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


def fill_small_gaps(vectors, door_detections):
    """
    Step 3: Wall Gap Normalization.
    Fill tiny noise gaps that have no door bbox nearby.
    """
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
            gaps_to_merge.add((i, j, ei_idx, ej_idx, gap_dist))
            gaps_filled += 1

    # Apply gap fills by extending endpoints to meet
    modified_vectors = list(vectors)

    for (i, j, ei_idx, ej_idx, gap_dist) in gaps_to_merge:
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
    print(f"  [2C-Result] {len(vectors)} input vectors → {len(modified_vectors)} output vectors")

    return modified_vectors


def carve_doors_out_of_walls(vectors, door_detections):
    """
    Slices the 2D wall graph vectors to create physical topological gaps where doors are detected.
    This fulfills the BIM requirement of having actual modeled openings natively.
    """
    valid_doors = [d for d in door_detections if d.get("status") == "valid_door"]
    modified_vectors = list(vectors)
    door_cuts = []
    
    doors_carved = 0

    for det in valid_doors:
        det_center = np.array([det["x_center"], det["y_center"]])
        
        import pipeline
        door_widths_m = {"single_door": 0.9, "sliding_door": 1.2, "double_door": 1.6}
        door_width_px = door_widths_m.get(det["class"], 0.9) / pipeline.PIXEL_TO_METER
        bbox_aspect = det["width"] / (det["height"] + 1e-5)
        # Determine door's dominant axis unit vector
        if bbox_aspect >= 1.0:
            door_axis = np.array([1.0, 0.0])  # horizontal door
        else:
            door_axis = np.array([0.0, 1.0])  # vertical door

        best_wall_idx = -1
        best_cost = float('inf')
        
        for vi, (p1, p2) in enumerate(modified_vectors):
            p1_np, p2_np = np.array(p1), np.array(p2)
            
            # Perpendicular distance component
            d_perp, _ = _point_to_segment_dist(det_center, p1_np, p2_np)
            
            wall_dir_v = p2_np - p1_np
            wall_len_v = np.linalg.norm(wall_dir_v)
            if wall_len_v < 1e-5:
                continue
            w_norm = wall_dir_v / wall_len_v
            
            # Orientation cost: 0 = perfectly aligned, 1 = perpendicular
            # J_orient = 1 - |dot(wall_dir, door_axis)|
            j_orient = 1.0 - abs(np.dot(w_norm, door_axis))
            
            # Projection constraint: penalise centers projecting outside the segment
            t = np.dot(det_center - p1_np, w_norm)
            half_gap = door_width_px / 2.0
            sigma_far = max(door_width_px, 20.0)
            overshoot = max(0, t - (wall_len_v + half_gap))
            undershoot = max(0, -half_gap - t)
            j_proj = (overshoot**2 + undershoot**2) / (sigma_far**2)
            
            # Combined cost — weights tuned to noise characteristics
            # α=1 (distance in px), β=40 (orientation, same scale as 40px penalty), γ=1 (projection)
            cost = d_perp + 40.0 * j_orient + j_proj
                    
            if cost < best_cost:
                best_cost = cost
                best_wall_idx = vi
                
        # We cap the reasonable bounding cost to standard max distance mapping heuristics
        if best_cost > MAX_DOOR_WALL_DIST * 2.0 or best_wall_idx == -1:
            continue
            
        p1, p2 = modified_vectors[best_wall_idx]
        p1_np, p2_np = np.array(p1), np.array(p2)
        wall_vec = p2_np - p1_np
        wall_len = np.linalg.norm(wall_vec)
        
        if wall_len < 5.0:
            continue
            
        wall_dir = wall_vec / wall_len
        proj_center = np.dot(det_center - p1_np, wall_dir)
        proj_min = max(0.0, proj_center - door_width_px / 2.0)
        proj_max = min(wall_len, proj_center + door_width_px / 2.0)
        
        # MathGPT Q3 fix: skip only if the entire wall is consumed
        # Never silent-skip based on small stub size alone
        if proj_max - proj_min < 2.0:
            continue   # gap narrower than 2px — wrong wall or bad projection
            
        cut_start_pt = p1_np + proj_min * wall_dir
        cut_end_pt   = p1_np + proj_max * wall_dir
        
        new_segments = []
        S_L = proj_min                   # length of left stub
        S_R = wall_len - proj_max        # length of right stub
        
        if S_L > 2.0:
            new_segments.append((tuple(p1_np), tuple(cut_start_pt)))
        # if S_L <= 2.0 → door is near p1, left stub absorbed — correct behaviour
            
        if S_R > 2.0:
            new_segments.append((tuple(cut_end_pt), tuple(p2_np)))
        # if S_R <= 2.0 → door is near p2, right stub absorbed — correct behaviour
        # Both <= 2.0 only if wall_len ≈ door_width, i.e. wall shorter than door — correct to remove it
            
        modified_vectors.pop(best_wall_idx)
        for seg in reversed(new_segments):
            modified_vectors.insert(best_wall_idx, seg)
            
        door_cuts.append((tuple(cut_start_pt), tuple(cut_end_pt), tuple(wall_dir)))
            
        doors_carved += 1
        
    print(f"  [2C-Carve] Carved {doors_carved} topological gaps into the walls for detected doors.")
    return modified_vectors, door_cuts

def correct_structure(image, vectors, weights_path="best.pt", is_fast_mode=False):
    """
    Main entry point for Phase 2C.
    Takes the original image and Phase 5B vectors, returns corrected vectors.
    """
    print("\n=== Phase 2C: Structural Correction ===")

    # 1. Load YOLO and run inference
    model = load_yolo_model(weights_path)
    detections = run_yolo_inference(model, image)
    print(f"  [2C] YOLO detected {len(detections)} objects")
    
    # 1.b Math scale for Fast Mode mismatch
    if is_fast_mode:
        h, w = image.shape[:2]
        scale = 512.0 / max(h, w)
        # Keep padding in float — don't truncate to int until final coordinate assignment
        new_w_f = w * scale
        new_h_f = h * scale
        pad_left_f = (512.0 - new_w_f) / 2.0
        pad_top_f  = (512.0 - new_h_f) / 2.0
        for det in detections:
            for k in ["x1", "x2", "x_center"]:
                det[k] = det[k] * scale + pad_left_f
            for k in ["y1", "y2", "y_center"]:
                det[k] = det[k] * scale + pad_top_f
            det["width"] *= scale
            det["height"] *= scale

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

    # 4. Fill Noise Gaps
    # Only use valid doors for gap interference
    door_dets = [d for d in detections if d.get("status") == "valid" and d["class"] in DOOR_CLASSES]
    
    # Tag them specifically so the UI knows these were the ones used for structural influence
    for d in door_dets:
        d["status"] = "valid_door"
        
    # Phase A: Close small fractures
    vectors_closed = fill_small_gaps(vectors, door_dets)
    
    # 5. Carve the valid doors out of the vector topology
    # We now also catch door_cuts to pass up to Babylon.js Phase 6 tracking
    vectors_phase2c, door_cuts = carve_doors_out_of_walls(vectors_closed, door_dets)
    
    return vectors_phase2c, detections, door_cuts


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
    vectors, _, _ = pipeline.process_geometry(mask)
    corrected, dets, cuts = correct_structure(img, vectors)
    print(f"\nOriginal: {len(vectors)} vectors → Corrected: {len(corrected)} vectors")
    print(f"Detections used: {len(dets)}, Door cuts: {len(cuts)}")
