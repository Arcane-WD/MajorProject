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
    # binary mask
    binary = (mask > 0.5).astype(np.uint8) * 255

    # Remove Small Blobs (Dust)
    nb_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1] # Skip background
    clean_mask = np.zeros_like(binary)
    for i in range(len(sizes)):
        if sizes[i] >= min_blob_size:
            clean_mask[labels == i + 1] = 255

    # 3. Morphological Closing (Bridge Gaps)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    # 4. Morphological Opening (Smooth Edges)
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
    
    # Draw the skeleton path with thickness = radius * 2
    # This creates the "Search Region"
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
            simple_path = rdp(path, epsilon=EPSILON)
            
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

def process_geometry(mask):
    # Phase 4B: Refine Mask (Research-Grade Cleaning)
    clean_mask = refine_mask(mask)
    
    # Skeletonize (Topology Only)
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
        # Rounding needed because PCA returns floats
        t_p1 = tuple(np.round(p1).astype(int))
        t_p2 = tuple(np.round(p2).astype(int))
        
        edge = tuple(sorted((t_p1, t_p2)))
        length = np.linalg.norm(np.array(p1)-np.array(p2))
        
        if edge not in seen and length > MIN_LENGTH:
            seen.add(edge)
            final_vectors.append((p1, p2))
            
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