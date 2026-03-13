import torch
import cv2
import numpy as np
import networkx as nx
import trimesh
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
from rdp import rdp
import itertools
import os
import math
from scipy.spatial import KDTree

# --- CONSTANTS ---
PIXEL_TO_METER = 0.05
BASE_RESOLUTION = 512
MAX_DIM = 4096
WALL_HEIGHT = 2.5
DOOR_HEIGHT = 2.1
HEADER_SIZE = WALL_HEIGHT - DOOR_HEIGHT
WALL_THICKNESS = 0.2
DOOR_WIDTH_MAX = 1.2
EPSILON = 2.0
MIN_LENGTH = 15

# --- PHASE 5B CONSTANTS (TUNABLE) ---
SNAP_THRESHOLD = 4       # pixels — cluster endpoints closer than this
ANGLE_TOLERANCE = 7.0    # degrees — snap to axis if within this tolerance
GAP_THRESHOLD = 10       # pixels — extend endpoints to nearby wall segments

def load_model_logic(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_weight_map(tile_size):
    window = np.hanning(tile_size)
    return np.outer(window, window)

def predict_tiled(model, device, image, tile_size=512, overlap=0.5, progress_callback=None):
    h, w = image.shape[:2]
    
    # Conditional Exit
    if h <= tile_size and w <= tile_size:
        if progress_callback: progress_callback(1.0)
        return predict_mask(model, device, image)
    
    # System throttling safety
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        h, w = new_h, new_w
    
    stride = int(tile_size * (1 - overlap))
    full_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)
    weight_map = get_weight_map(tile_size)
    
    y_starts = range(0, h, stride)
    x_starts = range(0, w, stride)
    total_tiles = len(y_starts) * len(x_starts)
    processed = 0
    
    for y in y_starts:
        for x in x_starts:
            y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
            y0, x0 = max(0, y1 - tile_size), max(0, x1 - tile_size)
            tile = image[y0:y1, x0:x1]
            
            pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            t = tile.astype(np.float32) / 255.0
            t = (t - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            t = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                pred = model(t).sigmoid().cpu().numpy().squeeze()
            
            vh, vw = tile_size - pad_h, tile_size - pad_w
            full_mask[y0:y1, x0:x1] += pred[:vh, :vw] * weight_map[:vh, :vw]
            count_mask[y0:y1, x0:x1] += weight_map[:vh, :vw]
            
            processed += 1
            if progress_callback: progress_callback(processed / total_tiles)

    np.place(count_mask, count_mask == 0, 1.0)
    return full_mask / count_mask

def predict_mask(model, device, image):
    h, w = image.shape[:2]
    scale = 512 / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(image, (nw, nh))
    
    dh, dw = 512 - nh, 512 - nw
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    t = img.astype('float32') / 255.0
    t = (t - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        return model(t).sigmoid().cpu().numpy().squeeze()


def refine_mask(mask, min_blob_size=50):
    """
    Cleans the probability mask using statistics and morphology.
    """
    binary = (mask > 0.7).astype(np.uint8) * 255

    nb_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1] # Skip background
    clean_mask = np.zeros_like(binary)
    for i in range(len(sizes)):
        if sizes[i] >= min_blob_size:
            clean_mask[labels == i + 1] = 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)

    return clean_mask


def extract_wall_pixels(path_points, clean_mask, radius=5):
    """
    Creates a search tunnel around the skeleton path and extracts
    all actual wall pixels from the clean mask within that tunnel.
    """
    mask_h, mask_w = clean_mask.shape
    temp_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    
    pts = np.array(path_points, dtype=np.int32)
    cv2.polylines(temp_mask, [pts], isClosed=False, color=255, thickness=radius*2)
    
    # Intersect with the actual wall mass
    wall_region = cv2.bitwise_and(temp_mask, clean_mask)
    
    # Extract coordinates of all non-zero pixels
    # numpy returns (row, col) -> (y, x). We flip to (x, y) for cv2 compatibility.
    y_locs, x_locs = np.where(wall_region > 0)
    pixel_cloud = np.column_stack((x_locs, y_locs))
    
    return pixel_cloud

def fit_line_to_cloud(pixel_cloud, start_point, end_point):
    """
    Fits a mathematical line (PCA/Least Squares) to a cloud of pixels.
    Projects the original skeleton start/end points onto this ideal line.
    """
    if len(pixel_cloud) < 5: 
        return start_point, end_point # Not enough data, return original
        
    # Fit line using Least Squares (DIST_L2)
    # Returns normalized vector (vx, vy) and a point on the line (x0, y0)
    line = cv2.fitLine(pixel_cloud, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = line[0][0], line[1][0]
    x0, y0 = line[2][0], line[3][0]
    
    # Helper to project a point (px, py) onto the fitted line
    def project(p):
        px, py = p
        # Vector from line origin to point
        vec_x, vec_y = px - x0, py - y0
        # Dot product with line direction
        dot = vec_x * vx + vec_y * vy
        # Projected point
        proj_x = x0 + dot * vx
        proj_y = y0 + dot * vy
        return (proj_x, proj_y)

    new_start = project(start_point)
    new_end = project(end_point)
    
    return new_start, new_end

def vectorize_hybrid(G, clean_mask):
    """
    Phase 5A: Skeleton-Guided Line Fitting (The Research-Grade Approach).
    1. Breaks skeleton paths into straight-ish segments using RDP.
    2. Collects actual wall pixels around those segments.
    3. Fits a mathematical line to the pixels (Region-to-Vector).
    """
    junctions = [n for n in G.nodes() if G.degree(n) != 2]
    if not junctions and len(G) > 0: junctions = [list(G.nodes())[0]]
    
    vectors = []
    G_temp = G.copy()
    
    for start in junctions:
        if start not in G_temp: continue
        
        neighbors = list(G_temp.neighbors(start))
        for next_node in neighbors:
            if not G_temp.has_edge(start, next_node): continue
            
            # Trace skeleton path
            path = [start, next_node]
            curr, prev = next_node, start
            while G.degree(curr) == 2:
                nbrs = [n for n in G_temp.neighbors(curr) if n != prev]
                if not nbrs: break
                prev, curr = curr, nbrs[0]
                path.append(curr)
                if curr in junctions: break
            
            # Remove from graph
            for i in range(len(path)-1):
                if G_temp.has_edge(path[i], path[i+1]): 
                    G_temp.remove_edge(path[i], path[i+1])
            
            
            # 1. Use RDP to find geometric breaks (corners that aren't junctions)
            # This turns curves/L-shapes into straight sub-segments
            simple_path = rdp(path, epsilon=EPSILON)#VYW method
            
            # 2. Process each sub-segment
            for i in range(len(simple_path) - 1):
                p_start = simple_path[i]
                p_end = simple_path[i+1]
                
                # Create a mini-segment list for pixel extraction
                segment_path = [p_start, p_end]
                
                # 3. Extract real wall pixels around this skeleton segment
                pixel_cloud = extract_wall_pixels(segment_path, clean_mask, radius=7)
                
                # 4. Fit a perfect line to the pixel cloud
                fitted_start, fitted_end = fit_line_to_cloud(pixel_cloud, p_start, p_end)
                
                vectors.append((fitted_start, fitted_end))
                
    return vectors

# ============================================================
# PHASE 5B: Junction & Topology Optimization
# ============================================================

def snap_vertices(vectors, threshold=SNAP_THRESHOLD):
    """
    Cluster-based vertex snapping using Union-Find.
    - Only merges endpoints within direct radius (no chain-clustering)
    - Both endpoints of the SAME wall are never merged together
      (prevents short walls from collapsing)
    """
    if not vectors:
        return vectors
    
    # 1. Collect all endpoints
    endpoints = []
    for p1, p2 in vectors:
        endpoints.append(np.array(p1, dtype=np.float64))
        endpoints.append(np.array(p2, dtype=np.float64))
    
    pts = np.array(endpoints)
    n = len(pts)
    
    # 2. Union-Find structure
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # 3. KD-tree pairwise queries — only direct neighbors, no BFS chaining
    tree = KDTree(pts)
    pairs = tree.query_pairs(threshold)
    
    # Build set of same-wall endpoint pairs (indices i*2 and i*2+1)
    same_wall_pairs = set()
    for i in range(len(vectors)):
        same_wall_pairs.add((i * 2, i * 2 + 1))
        same_wall_pairs.add((i * 2 + 1, i * 2))
    
    for (a, b) in pairs:
        # Never merge endpoints of the same wall
        if (a, b) in same_wall_pairs:
            continue
        union(a, b)
    
    # 4. Group by cluster root and compute centroids
    from collections import defaultdict
    cluster_map = defaultdict(list)
    for i in range(n):
        cluster_map[find(i)].append(i)
    
    point_map = {}
    snapped_count = 0
    for root, members in cluster_map.items():
        centroid = np.mean(pts[members], axis=0)
        for idx in members:
            point_map[idx] = centroid
        if len(members) > 1:
            snapped_count += len(members)
    
    # 5. Rebuild vectors with snapped endpoints
    snapped_vectors = []
    for i, (p1, p2) in enumerate(vectors):
        new_p1 = tuple(point_map[i * 2])
        new_p2 = tuple(point_map[i * 2 + 1])
        # Skip only if truly collapsed (< 1px)
        if np.linalg.norm(np.array(new_p1) - np.array(new_p2)) > 1.0:
            snapped_vectors.append((new_p1, new_p2))
    
    multi_clusters = sum(1 for c in cluster_map.values() if len(c) > 1)
    print(f"  [5B-Snap] {snapped_count} endpoints snapped across {multi_clusters} clusters (threshold={threshold}px)")
    print(f"  [5B-Snap] {len(vectors)} -> {len(snapped_vectors)} vectors ({len(vectors) - len(snapped_vectors)} collapsed)")
    return snapped_vectors


def enforce_manhattan(vectors, angle_tol=ANGLE_TOLERANCE):
    """
    Manhattan-world enforcement.
    Walls nearly aligned with 0/90/180/270 degrees are snapped to
    exact axis alignment. Midpoint is preserved.
    """
    if not vectors:
        return vectors
    
    corrected = []
    corrections = 0
    
    for p1, p2 in vectors:
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        
        if length < 1.0:
            continue
        
        # Find nearest cardinal direction (0, 90, 180, -90)
        cardinal_angles = [0, 90, 180, -180, -90]
        min_diff = float('inf')
        snap_angle = angle_deg
        
        for ca in cardinal_angles:
            diff = abs(angle_deg - ca)
            if diff < min_diff:
                min_diff = diff
                snap_angle = ca
        
        if min_diff <= angle_tol:
            # Snap: rotate around midpoint
            mid = (p1 + p2) / 2.0
            half_len = length / 2.0
            snap_rad = math.radians(snap_angle)
            
            new_p1 = mid - half_len * np.array([math.cos(snap_rad), math.sin(snap_rad)])
            new_p2 = mid + half_len * np.array([math.cos(snap_rad), math.sin(snap_rad)])
            corrected.append((tuple(new_p1), tuple(new_p2)))
            corrections += 1
        else:
            corrected.append((tuple(p1), tuple(p2)))
    
    print(f"  [5B-Manhattan] {corrections}/{len(vectors)} walls snapped to axis")
    return corrected


def close_gaps(vectors, gap_threshold=GAP_THRESHOLD):
    """
    T-junction gap closure.
    For endpoints not already connected to another endpoint,
    find the nearest wall segment and extend to meet it.
    """
    if not vectors:
        return vectors
    
    def point_to_segment_dist(p, a, b):
        """Distance from point p to segment ab, and the closest point on ab."""
        ap = p - a
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-10:
            return np.linalg.norm(ap), a
        t = np.dot(ap, ab) / ab_sq
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest
    
    # Collect all endpoints and check which are "free" (not shared)
    endpoint_counts = {}  # rounded coord -> count
    for p1, p2 in vectors:
        k1 = (round(p1[0], 1), round(p1[1], 1))
        k2 = (round(p2[0], 1), round(p2[1], 1))
        endpoint_counts[k1] = endpoint_counts.get(k1, 0) + 1
        endpoint_counts[k2] = endpoint_counts.get(k2, 0) + 1
    
    closed_vectors = list(vectors)  # mutable copy
    closures = 0
    
    for vi in range(len(closed_vectors)):
        p1, p2 = closed_vectors[vi]
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        
        for endpoint_idx, ep in enumerate([p1, p2]):
            ep_key = (round(ep[0], 1), round(ep[1], 1))
            
            # Only process "free" endpoints (connected to only 1 wall)
            if endpoint_counts.get(ep_key, 0) > 1:
                continue
            
            # Find nearest segment (excluding own segment)
            best_dist = float('inf')
            best_point = None
            
            for vj in range(len(closed_vectors)):
                if vi == vj:
                    continue
                a = np.array(closed_vectors[vj][0], dtype=np.float64)
                b = np.array(closed_vectors[vj][1], dtype=np.float64)
                dist, closest = point_to_segment_dist(ep, a, b)
                
                if dist < best_dist:
                    best_dist = dist
                    best_point = closest
            
            if best_dist < gap_threshold and best_point is not None:
                # Extend this endpoint to the closest point on the nearest segment
                if endpoint_idx == 0:
                    closed_vectors[vi] = (tuple(best_point), tuple(p2))
                else:
                    closed_vectors[vi] = (tuple(p1), tuple(best_point))
                closures += 1
    
    print(f"  [5B-GapClose] {closures} endpoints extended to nearby walls")
    return closed_vectors


def process_geometry(mask):
    # Phase 4B: Refine Mask (Research-Grade Cleaning)
    clean_mask = refine_mask(mask)
    
    # Skeletonize (Topology Only)
    #threshold variation
    skeleton = skeletonize(clean_mask > 0).astype(np.uint8)
    
    # Graph Construction
    y, x = np.where(skeleton > 0)
    points = list(zip(x, y))
    points_set = set(points)
    G = nx.Graph()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for u, v in points:
        G.add_node((u, v))
        for du, dv in shifts:
            neighbor = (u + du, v + dv)
            if neighbor in points_set and (u, v) < neighbor:
                G.add_edge((u, v), neighbor, weight=1.0)
    
    # Phase 5A: Hybrid Vectorization (Geometry from Pixels, Topology from Skeleton)
    # Note: We pass clean_mask now!
    vectors = vectorize_hybrid(G, clean_mask)
    
    # Post-Vector Pruning
    final_vectors = []
    seen = set()
    for p1, p2 in vectors:
        #optimize O(nlogn)
        # Rounding needed because PCA returns floats
        t_p1 = tuple(np.round(p1).astype(int))
        t_p2 = tuple(np.round(p2).astype(int))
        
        edge = tuple(sorted((t_p1, t_p2)))
        length = np.linalg.norm(np.array(p1)-np.array(p2))
        
        if edge not in seen and length > MIN_LENGTH:
            seen.add(edge)
            final_vectors.append((p1, p2))
    
    # Phase 5B: Junction & Topology Optimization
    print(f"  [5A] {len(final_vectors)} raw vectors extracted")
    final_vectors = snap_vertices(final_vectors)
    final_vectors = enforce_manhattan(final_vectors)
    final_vectors = close_gaps(final_vectors)
    print(f"  [5B] {len(final_vectors)} vectors after topology optimization")
            
    return final_vectors

def create_box(p1, p2, thickness, height, z_offset=0):
    p1, p2 = np.array(p1), np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 0.01: return None
    angle = np.arctan2(direction[1], direction[0])
    box = trimesh.creation.box(extents=[length, thickness, height])
    box.apply_translation([length/2, 0, 0])
    box.apply_transform(trimesh.transformations.rotation_matrix(angle, [0,0,1]))
    box.apply_translation([p1[0], p1[1], z_offset + height/2])
    return box
#change next phase
def are_collinear(p1, p2, p3, p4, tol=0.95):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return False
    return abs(np.dot(v1, v2) / (n1 * n2)) > tol

def generate_3d_scene(vectors):
    scene_meshes = []
    scaled_vectors = []
    
    # Walls
    for p1, p2 in vectors:
        s_p1 = np.array(p1) * PIXEL_TO_METER
        s_p2 = np.array(p2) * PIXEL_TO_METER
        scaled_vectors.append((s_p1, s_p2))
        wall = create_box(s_p1, s_p2, WALL_THICKNESS, WALL_HEIGHT)
        if wall:
            wall.visual.face_colors = [220, 220, 220, 255]
            scene_meshes.append(wall)
            
    # Headers
    for i, j in itertools.combinations(range(len(scaled_vectors)), 2):
        w1_s, w1_e = scaled_vectors[i]
        w2_s, w2_e = scaled_vectors[j]
        pairs = [(w1_s, w2_s), (w1_s, w2_e), (w1_e, w2_s), (w1_e, w2_e)]
        best_pair, min_dist = None, float('inf')
        for p1, p2 in pairs:
            d = np.linalg.norm(np.array(p1)-np.array(p2))
            if d < min_dist: min_dist, best_pair = d, (p1, p2)
        #make dynamic
        if 0.6 < min_dist < DOOR_WIDTH_MAX:
            orig_w1_s, orig_w1_e = vectors[i]
            orig_w2_s, orig_w2_e = vectors[j]
            if are_collinear(orig_w1_s, orig_w1_e, orig_w2_s, orig_w2_e):
                header = create_box(best_pair[0], best_pair[1], WALL_THICKNESS, HEADER_SIZE, z_offset=DOOR_HEIGHT)
                if header:
                    header.visual.face_colors = [200, 200, 200, 255]
                    scene_meshes.append(header)
    
    # Floor
    if scaled_vectors:
        pts = np.array([p for pair in scaled_vectors for p in pair])
        min_x, min_y, max_x, max_y = pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()
        w, d = (max_x - min_x) * 1.2, (max_y - min_y) * 1.2
        cx, cy = (max_x + min_x)/2, (max_y + min_y)/2
        floor = trimesh.creation.box(extents=[w, d, 0.2])
        floor.apply_translation([cx, cy, -0.1])
        floor.visual.face_colors = [100, 100, 100, 255]
        scene_meshes.append(floor)

    if not scene_meshes: return None

    combined_mesh = trimesh.util.concatenate(scene_meshes)
    rotation = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    combined_mesh.apply_transform(rotation)
    return combined_mesh