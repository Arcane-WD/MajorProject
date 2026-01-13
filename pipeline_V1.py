# --- 1. INSTALL DEPENDENCIES (If needed) ---
# !pip install segmentation-models-pytorch rdp trimesh networkx

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
import networkx as nx
import trimesh
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
from rdp import rdp

# --- 2. CONFIGURATION & CONSTANTS ---
MODEL_PATH = "best_cleaner_model.pth"
PROCESSED_DIR = 'processed_dataset_512'
OUTPUT_DIR = 'batch_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architectural Standards (Meters)
PIXEL_TO_METER = 0.05        # 1 px = 5cm
WALL_HEIGHT = 2.5            # Standard Ceiling
DOOR_HEIGHT = 2.1            # Door Header Height
HEADER_SIZE = WALL_HEIGHT - DOOR_HEIGHT
WALL_THICKNESS = 0.2         # 20cm walls
DOOR_WIDTH_MAX = 1.2         # Max width for a door vs archway
EPSILON = 2.0                # RDP tolerance
MIN_LENGTH = 15              # Prune walls shorter than 15px

# --- 3. MODEL LOADER ---
print(f"Loading Model on {device}...")
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3, 
    classes=1
)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found.")

# --- 4. PHASE 1: PERCEPTION (Optimized Inference) ---
def predict_image(image_path):
    # Load
    original_img = cv2.imread(image_path)
    if original_img is None: return None, None
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Optimization A: Aspect Ratio Preservation (Pad instead of Stretch)
    h, w = original_img.shape[:2]
    scale = 512 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(original_img, (new_w, new_h))
    
    # Pad to 512x512
    delta_w = 512 - new_w
    delta_h = 512 - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    
    # Normalize
    input_tensor = padded_img.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_tensor = (input_tensor - mean) / std
    
    # Inference
    input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pr_mask = logits.sigmoid().cpu().numpy().squeeze()
        
    return padded_img, pr_mask

# --- 5. PHASE 2: DRAFTING (Optimized Geometry) ---
def build_graph(skeleton):
    y, x = np.where(skeleton > 0)
    points = list(zip(x, y))
    points_set = set(points)
    G = nx.Graph()
    
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Optimization D: Faster Graph Build (Avoid double edge checks)
    for u, v in points:
        G.add_node((u, v))
        for du, dv in shifts:
            neighbor = (u + du, v + dv)
            if neighbor in points_set:
                # Canonical check to add edge only once
                if (u, v) < neighbor: 
                    G.add_edge((u, v), neighbor, weight=1.0)
    return G

def vectorize(G):
    # Detect Junctions
    junctions = [n for n in G.nodes() if G.degree(n) != 2 and G.degree(n) > 0]
    if not junctions and G.number_of_nodes() > 0:
        junctions = [list(G.nodes())[0]] 
        
    vectors = []
    G_temp = G.copy()
    
    for start in junctions:
        if start not in G_temp: continue
            
        neighbors = list(G_temp.neighbors(start))
        for next_node in neighbors:
            if not G_temp.has_edge(start, next_node): continue
            
            # Trace Path
            path = [start, next_node]
            curr = next_node
            prev = start
            
            while G.degree(curr) == 2:
                nbrs = [n for n in G_temp.neighbors(curr) if n != prev]
                if not nbrs: break
                next_step = nbrs[0]
                path.append(next_step)
                prev = curr
                curr = next_step
                if curr in junctions: break
            
            # Remove traced edges
            for i in range(len(path) - 1):
                if G_temp.has_edge(path[i], path[i+1]):
                    G_temp.remove_edge(path[i], path[i+1])
            
            # RDP Simplification
            if len(path) > 2:
                simple = rdp(path, epsilon=EPSILON)
                for i in range(len(simple) - 1):
                    vectors.append((simple[i], simple[i+1]))
    return vectors

# --- 6. PHASE 3: CONSTRUCTION (Safety-Railed Logic) ---
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
    """Optimization F: Check if two segments are roughly collinear."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return False
    
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    return abs(cosine) > tol

def build_headers(vectors, wall_meshes):
    header_meshes = []
    
    # We need to look at WALL pairs, not just endpoints, to check alignment
    # Structure: vectors[i] is (start, end)
    for i, j in itertools.combinations(range(len(vectors)), 2):
        w1_start, w1_end = vectors[i]
        w2_start, w2_end = vectors[j]
        
        # Check all 4 connection possibilities (start-start, start-end, etc)
        # We want the shortest distance between any endpoint of Wall A and Wall B
        pairs = [
            (w1_start, w2_start), (w1_start, w2_end),
            (w1_end, w2_start),   (w1_end, w2_end)
        ]
        
        valid_header = False
        best_pair = None
        min_dist = float('inf')
        
        for p1, p2 in pairs:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < min_dist:
                min_dist = dist
                best_pair = (p1, p2)
        
        # Criteria 1: Door-sized gap?
        if 0.6 < min_dist < DOOR_WIDTH_MAX:
            # Criteria 2 (Optimization F): Are walls collinear?
            if are_collinear(w1_start, w1_end, w2_start, w2_end):
                header = create_box(best_pair[0], best_pair[1], WALL_THICKNESS, HEADER_SIZE, z_offset=DOOR_HEIGHT)
                if header:
                    header.visual.face_colors = [200, 200, 200, 255]
                    header_meshes.append(header)

    return header_meshes

# --- 7. PIPELINE ORCHESTRATOR (Batch-Ready) ---
def run_full_pipeline(image_path, filename):
    print(f"\nProcessing V1.1: {filename}...")
    
    # A. Perception
    original_img, mask = predict_image(image_path)
    if mask is None: return

    # B. Drafting
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Optimization B: Safer Morphology
    # Use smaller kernel for standard 512px to keep details
    kernel = np.ones((3, 3), np.uint8) 
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    skeleton = skeletonize(closed_mask > 0).astype(np.uint8)
    G = build_graph(skeleton)
    raw_vectors = vectorize(G)
    
    # Pruning & Deduplication
    final_vectors = []
    seen = set()
    for p1, p2 in raw_vectors:
        edge = tuple(sorted((tuple(p1), tuple(p2))))
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        if edge not in seen and length > MIN_LENGTH:
            seen.add(edge)
            final_vectors.append((p1, p2))
            
    # C. Construction
    scene_meshes = []
    scaled_vectors = []
    
    # Walls
    for p1, p2 in final_vectors:
        s_p1 = np.array(p1) * PIXEL_TO_METER
        s_p2 = np.array(p2) * PIXEL_TO_METER
        scaled_vectors.append((s_p1, s_p2))
        wall = create_box(s_p1, s_p2, WALL_THICKNESS, WALL_HEIGHT)
        if wall:
            wall.visual.face_colors = [220, 220, 220, 255]
            scene_meshes.append(wall)
            
    # Headers
    headers = build_headers(scaled_vectors, scene_meshes)
    scene_meshes.extend(headers)
    
    # Floor Slab
    if scaled_vectors:
        all_points = np.array([p for pair in scaled_vectors for p in pair])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        w, d = (max_x - min_x) * 1.2, (max_y - min_y) * 1.2
        cx, cy = (max_x + min_x)/2, (max_y + min_y)/2
        
        floor = trimesh.creation.box(extents=[w, d, 0.2])
        floor.apply_translation([cx, cy, -0.1])
        floor.visual.face_colors = [100, 100, 100, 255]
        scene_meshes.append(floor)

    # D. Export
    if scene_meshes:
        save_path = os.path.join(OUTPUT_DIR, f"{filename}.glb")
        combined = trimesh.util.concatenate(scene_meshes)
        combined.export(save_path)
        print(f"✅ Saved {filename}.glb (Vectors: {len(final_vectors)}, Headers: {len(headers)})")
    else:
        print(f"❌ Failed to generate geometry for {filename}")

# --- 8. EXECUTION ---
if __name__ == "__main__":
    images_dir = os.path.join(PROCESSED_DIR, "images")
    if os.path.exists(images_dir):
        all_files = os.listdir(images_dir)
        BATCH_SIZE = 5
        sample_files = random.sample(all_files, min(BATCH_SIZE, len(all_files)))
        
        print(f"🚀 Starting V1.1 Pipeline on {len(sample_files)} images...")
        for fname in sample_files:
            try:
                run_full_pipeline(os.path.join(images_dir, fname), fname)
            except Exception as e:
                print(f"🔥 Error on {fname}: {e}")
        
        print(f"\n✨ Batch Complete! Check the '{OUTPUT_DIR}' folder.")
