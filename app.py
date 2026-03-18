import streamlit as st
import numpy as np
import cv2
import torch
import tempfile
import base64
import os
from skimage.morphology import skeletonize

import pipeline
import structural_corrector

# --- CONFIG ---
st.set_page_config(page_title="Scan-to-BIM Engine", page_icon="🏗️", layout="wide")

# --- CACHED MODEL ---
@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_cleaner_model_v3.pth"
    try:
        model = pipeline.load_model_logic(MODEL_PATH, device)
        return model, device
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None

# --- VIEWER ---
def render_3d_viewer(glb_bytes, height=500):
    b64 = base64.b64encode(glb_bytes).decode('utf-8')
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <style>
          model-viewer {{ width: 100%; height: {height}px; background-color: #1a1a2e; border-radius: 10px; }}
        </style>
      </head>
      <body>
        <model-viewer 
          src="data:model/gltf-binary;base64,{b64}" 
          alt="3D Floorplan" 
          camera-controls 
          auto-rotate 
          shadow-intensity="1" 
          camera-orbit="0deg 75deg 105%" 
          min-camera-orbit="-infinitydeg 0deg auto" 
          max-camera-orbit="infinitydeg 180deg auto">
        </model-viewer>
      </body>
    </html>
    """
    st.components.v1.html(html_code, height=height)

# --- HELPERS ---
def draw_vectors_on_image(shape, vectors, color=(0, 255, 0), thickness=2):
    canvas = np.zeros((*shape[:2], 3), dtype=np.uint8)
    for p1, p2 in vectors:
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(canvas, pt1, pt2, color, thickness)
    return canvas

def draw_yolo_boxes(image, detections):
    overlay = image.copy()
    colors = {
        "single_door": (0, 255, 0), "double_door": (255, 255, 0),
        "sliding_door": (0, 255, 255), "window": (255, 0, 0),
        "bay_window": (255, 128, 0), "blind_window": (128, 0, 255),
    }
    for det in detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        c = colors.get(det["class"], (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        label = f'{det["class"]} {det["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw, y1), c, -1)
        cv2.putText(overlay, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return overlay

def mesh_to_glb_bytes(mesh):
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        mesh.export(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()

# --- MAIN ---
def main():
    st.title("🏗️ Scan-to-BIM: AI Floorplan Reconstructor")
    
    # Sidebar
    st.sidebar.header("System Config")
    model, device = get_model()
    if not model: st.stop()
    st.sidebar.success(f"Engine Online ({device})")
    
    # Pipeline Mode
    pipeline_mode = st.sidebar.radio(
        "Pipeline Mode",
        ["⚡ Direct", "🔬 Demonstrative"],
        help="Direct = fast final output only. Demonstrative = step-by-step with all intermediates."
    )
    is_demo = pipeline_mode.startswith("🔬")
    
    # Inference Mode
    inference_mode = st.sidebar.radio("Inference Mode", ["Fast (512px)", "High Fidelity (Tiled)"])
    
    if inference_mode == "Fast (512px)":
        st.sidebar.caption("⚠️ **Preview Only:** Non-metric scale.")
    else:
        st.sidebar.caption("✅ **Metric Mode:** Preserves original scale.")

    uploaded_file = st.sidebar.file_uploader("Upload Floor Plan", type=["png", "jpg", "jpeg"])

    yolo_available = os.path.exists("best.pt")
    if yolo_available:
        st.sidebar.success("🎯 YOLOv8 weights detected")
    else:
        st.sidebar.warning("⚠️ No best.pt — Phase 2C disabled")

    if not uploaded_file:
        st.info("👈 Upload a floor plan image to begin")
        return

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ============================
    #  DIRECT MODE
    # ============================
    if not is_demo:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_rgb, caption=f"Input ({image.shape[1]}×{image.shape[0]})", width="stretch")
            run_btn = st.button("Generate 3D Model", type="primary", width="stretch")

        if run_btn:
            # Perception
            if inference_mode == "Fast (512px)":
                with st.spinner("Running Fast Perception..."):
                    mask = pipeline.predict_mask(model, device, image_rgb)
            else:
                progress_bar = st.progress(0, text="Running High-Fidelity Tiled Inference...")
                def update_progress(p):
                    progress_bar.progress(p, text=f"Stitching Tiles: {int(p*100)}%")
                mask = pipeline.predict_tiled(model, device, image_rgb, progress_callback=update_progress)
                progress_bar.empty()

            # Geometry + optional structural correction
            with st.spinner("Extracting Geometry..."):
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if yolo_available else None
                yolo_w = "best.pt" if yolo_available else None
                vectors, detections = pipeline.process_geometry(mask, original_image=bgr, yolo_weights=yolo_w)

            # 3D
            with st.spinner("Constructing 3D Model..."):
                mesh = pipeline.generate_3d_scene(vectors)

            if mesh:
                glb_bytes = mesh_to_glb_bytes(mesh)
                with col2:
                    det_msg = f" · {len(detections)} icons detected" if detections else ""
                    st.success(f"✅ {len(vectors)} walls reconstructed{det_msg}")
                    render_3d_viewer(glb_bytes)
                    st.download_button("📥 Download GLB", glb_bytes, "floorplan.glb", "model/gltf-binary")
            else:
                st.error("No geometry detected.")

    # ============================
    #  DEMONSTRATIVE MODE
    # ============================
    else:
        st.image(image_rgb, caption=f"Input ({image.shape[1]}×{image.shape[0]})", width="stretch")
        run_btn = st.button("🔬 Run Full Demonstration Pipeline", type="primary", width="stretch")

        if run_btn:
            import networkx as nx

            st.divider()
            st.header("Pipeline Stages")

            # ── Stage 1: Mask Prediction ──
            with st.status("Stage 1: Wall Mask Prediction", expanded=True) as status:
                if inference_mode == "Fast (512px)":
                    mask = pipeline.predict_mask(model, device, image_rgb)
                else:
                    mask = pipeline.predict_tiled(model, device, image_rgb, progress_callback=lambda p: None)

                mask_vis = (mask * 255).astype(np.uint8)
                clean_mask = pipeline.refine_mask(mask)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(mask_vis, caption="Raw U-Net Probability Mask", width="stretch", clamp=True)
                with c2:
                    st.image(clean_mask, caption="Refined Binary Mask", width="stretch", clamp=True)
                status.update(label="Stage 1: Complete ✅", state="complete")

            # ── Stage 2: Skeletonization ──
            with st.status("Stage 2: Topology Extraction", expanded=True) as status:
                skeleton = skeletonize(clean_mask > 0).astype(np.uint8) * 255
                st.image(skeleton, caption="Skeleton Topology", width="stretch", clamp=True)
                status.update(label="Stage 2: Complete ✅", state="complete")

            # ── Stage 3: Vectorization ──
            with st.status("Stage 3: Vectorization (5A + 5B)", expanded=True) as status:
                # Build graph
                skel_bin = skeleton // 255
                y, x = np.where(skel_bin > 0)
                points = list(zip(x.tolist(), y.tolist()))
                points_set = set(points)
                G = nx.Graph()
                for u, v in points:
                    G.add_node((u, v))
                    for du, dv in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nb = (u+du, v+dv)
                        if nb in points_set and (u,v) < nb:
                            G.add_edge((u,v), nb, weight=1.0)

                # Raw vectors
                vectors_raw = pipeline.vectorize_hybrid(G, clean_mask)
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

                # Phase 5B
                vectors_5b = pipeline.snap_vertices(list(final_raw))
                vectors_5b = pipeline.enforce_manhattan(vectors_5b)
                vectors_5b = pipeline.close_gaps(vectors_5b)

                c1, c2 = st.columns(2)
                with c1:
                    pre5b_img = draw_vectors_on_image(image.shape, final_raw, color=(0, 200, 255))
                    st.image(cv2.cvtColor(pre5b_img, cv2.COLOR_BGR2RGB),
                             caption=f"Pre-5B Vectors ({len(final_raw)} segments)", width="stretch")
                with c2:
                    post5b_img = draw_vectors_on_image(image.shape, vectors_5b, color=(0, 255, 0))
                    st.image(cv2.cvtColor(post5b_img, cv2.COLOR_BGR2RGB),
                             caption=f"Post-5B Vectors ({len(vectors_5b)} segments)", width="stretch")
                status.update(label="Stage 3: Complete ✅", state="complete")

            # ── Stage 4: YOLO Detection + Structural Correction ──
            vectors_final = vectors_5b
            detections = []

            if yolo_available:
                with st.status("Stage 4: YOLOv8 Detection & Structural Correction", expanded=True) as status:
                    vectors_corrected, all_tracked_dets = structural_corrector.correct_structure(image, vectors_5b)
                    
                    discarded_doors = [d for d in all_tracked_dets if d.get("status") == "discarded" and d["class"] in structural_corrector.DOOR_CLASSES]
                    valid_doors = [d for d in all_tracked_dets if d.get("status") == "valid_door"]

                    c1_yolo, c2_yolo = st.columns(2)
                    with c1_yolo:
                        discard_vis = draw_yolo_boxes(image_rgb, discarded_doors)
                        st.image(discard_vis, caption=f"Discarded Doors ({len(discarded_doors)})", width="stretch")
                    with c2_yolo:
                        valid_vis = draw_yolo_boxes(image_rgb, valid_doors)
                        st.image(valid_vis, caption=f"Used Doors ({len(valid_doors)})", width="stretch")

                    vectors_final = vectors_corrected

                    c1, c2 = st.columns(2)
                    with c1:
                        pre_corr_img = draw_vectors_on_image(image.shape, vectors_5b, color=(0, 255, 0))
                        st.image(cv2.cvtColor(pre_corr_img, cv2.COLOR_BGR2RGB),
                                 caption=f"Before Correction ({len(vectors_5b)})", width="stretch")
                    with c2:
                        post_corr_img = draw_vectors_on_image(image.shape, vectors_corrected, color=(0, 255, 128))
                        st.image(cv2.cvtColor(post_corr_img, cv2.COLOR_BGR2RGB),
                                 caption=f"After Correction ({len(vectors_corrected)})", width="stretch")

                    status.update(label="Stage 4: Complete ✅", state="complete")
            else:
                st.warning("⚠️ best.pt not found — Skipping Phase 2C structural correction")

            # ── Stage 5: 3D Model Generation ──
            with st.status("Stage 5: 3D Model Construction", expanded=True) as status:
                mesh_pre = pipeline.generate_3d_scene(vectors_5b)
                mesh_post = pipeline.generate_3d_scene(vectors_final)

                if mesh_pre and mesh_post and yolo_available:
                    glb_pre = mesh_to_glb_bytes(mesh_pre)
                    glb_post = mesh_to_glb_bytes(mesh_post)

                    st.subheader("Pre-Correction Model")
                    render_3d_viewer(glb_pre, height=450)
                    st.download_button("📥 Download Pre-Correction GLB", glb_pre,
                                       "floorplan_pre.glb", "model/gltf-binary", key="dl_pre")

                    st.subheader("Post-Correction Model")
                    render_3d_viewer(glb_post, height=450)
                    st.download_button("📥 Download Post-Correction GLB", glb_post,
                                       "floorplan_post.glb", "model/gltf-binary", key="dl_post")
                elif mesh_post:
                    glb_post = mesh_to_glb_bytes(mesh_post)
                    render_3d_viewer(glb_post)
                    st.download_button("📥 Download GLB", glb_post,
                                       "floorplan.glb", "model/gltf-binary")
                else:
                    st.error("No geometry detected.")

                status.update(label="Stage 5: Complete ✅", state="complete")

            # ── Summary ──
            st.divider()
            st.success(
                f"**Pipeline Complete!** "
                f"{len(final_raw)} raw → {len(vectors_5b)} post-5B → {len(vectors_final)} final vectors"
                + (f" · {len(detections)} YOLO detections used" if detections else "")
            )


if __name__ == "__main__":
    main()