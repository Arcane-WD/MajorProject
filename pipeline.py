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

# --- CONSTANTS ---
PIXEL_TO_METER = 0.05
WALL_HEIGHT = 2.5
DOOR_HEIGHT = 2.1
HEADER_SIZE = WALL_HEIGHT - DOOR_HEIGHT
WALL_THICKNESS = 0.2
DOOR_WIDTH_MAX = 1.2
EPSILON = 2.0
MIN_LENGTH = 15

def load_model_logic(model_path, device):
    """Loads the model architecture and weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_mask(model, device, image):
    """Phase 1: Perception - Raw Inference with Aspect Ratio Padding."""
    # Preprocess with padding
    h, w = image.shape[:2]
    scale = 512 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h))
    
    delta_w = 512 - new_w
    delta_h = 512 - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    input_tensor = padded_img.astype('float32') / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    input_tensor = (input_tensor - mean) / std
    input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        mask = logits.sigmoid().cpu().numpy().squeeze()
    return mask

def process_geometry(mask):
    """Phase 2: Drafting - Skeletonization & Vectorization."""
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    # Adaptive morphology
    k_size = 3
    kernel = np.ones((k_size, k_size), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(closed_mask > 0).astype(np.uint8)
    
    # Graph Build
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
    
    # Vectorize
    junctions = [n for n in G.nodes() if G.degree(n) != 2 and G.degree(n) > 0]
    if not junctions and len(G) > 0: junctions = [list(G.nodes())[0]]
    
    vectors = []
    G_temp = G.copy()
    for start in junctions:
        if start not in G_temp: continue
        for next_node in list(G_temp.neighbors(start)):
            if not G_temp.has_edge(start, next_node): continue
            path = [start, next_node]
            curr, prev = next_node, start
            while G.degree(curr) == 2:
                nbrs = [n for n in G_temp.neighbors(curr) if n != prev]
                if not nbrs: break
                prev, curr = curr, nbrs[0]
                path.append(curr)
                if curr in junctions: break
            
            for i in range(len(path)-1):
                if G_temp.has_edge(path[i], path[i+1]): G_temp.remove_edge(path[i], path[i+1])
            
            if len(path) > 2:
                simple = rdp(path, epsilon=EPSILON)
                for i in range(len(simple)-1): vectors.append((simple[i], simple[i+1]))
    
    # Prune
    final_vectors = []
    seen = set()
    for p1, p2 in vectors:
        edge = tuple(sorted((tuple(p1), tuple(p2))))
        if edge not in seen and np.linalg.norm(np.array(p1)-np.array(p2)) > MIN_LENGTH:
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
    """Phase 3: Construction - Walls, Headers, and Floor."""
    scene_meshes = []
    scaled_vectors = []
    
    # Walls
    for p1, p2 in vectors:
        s_p1, s_p2 = np.array(p1) * PIXEL_TO_METER, np.array(p2) * PIXEL_TO_METER
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
             # Check collinearity using original unscaled vectors
            orig_w1_s, orig_w1_e = vectors[i]
            orig_w2_s, orig_w2_e = vectors[j]
            if are_collinear(orig_w1_s, orig_w1_e, orig_w2_s, orig_w2_e):
                header = create_box(best_pair[0], best_pair[1], WALL_THICKNESS, HEADER_SIZE, z_offset=DOOR_HEIGHT)
                if header:
                    header.visual.face_colors = [200, 200, 200, 255]
                    scene_meshes.append(header)
    
    # ... (previous code inside generate_3d_scene) ...
    
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

    if not scene_meshes:
        return None

    # --- FIX: ROTATE TO Y-UP FOR WEB VIEWING ---
    combined_mesh = trimesh.util.concatenate(scene_meshes)
    
    # Rotate -90 degrees around the X-axis
    rotation = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    combined_mesh.apply_transform(rotation)
    
    return combined_mesh