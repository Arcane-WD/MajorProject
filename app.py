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
    import streamlit.components.v1 as components
    components.html(html_code, height=height)

def render_navigation_viewer(glb_bytes, door_metadata, wall_vectors_px, height=700):
    import json
    
    glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")
    door_json = json.dumps(door_metadata)
    
    walls_json = json.dumps([
        [[float(p1[0] * pipeline.PIXEL_TO_METER), float(-p1[1] * pipeline.PIXEL_TO_METER)],
         [float(p2[0] * pipeline.PIXEL_TO_METER), float(-p2[1] * pipeline.PIXEL_TO_METER)]]
        for p1, p2 in wall_vectors_px
    ])
    
    # Calculate Spawn Point (Centroid of all wall endpoints)
    if wall_vectors_px:
        pts = np.array([p for pair in wall_vectors_px for p in pair]) * pipeline.PIXEL_TO_METER
        start_x, start_z = float(np.mean(pts[:, 0])), float(-np.mean(pts[:, 1]))
    else:
        start_x, start_z = 0.0, 0.0
        
    html = f"""
    <script src="https://cdn.babylonjs.com/babylon.js"></script>
    <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
    
    <style>
        #container {{ position: relative; width: 100%; height: {height}px; 
                      background: #1a1a2e; overflow: hidden; border-radius: 10px; }}
        #renderCanvas {{ width: 100%; height: 100%; touch-action: none; outline: none; cursor: crosshair; }}
        #minimap {{ position: absolute; top: 12px; right: 12px; 
                   width: 200px; height: 200px; background: rgba(0,0,0,0.5);
                   border: 1px solid rgba(255,255,255,0.3);
                   border-radius: 4px; pointer-events: none; }}
        #hud {{ position: absolute; top: 50%; left: 50%; 
                transform: translate(-50%, -50%);
                color: rgba(255,255,255,0.8); pointer-events: none; }}
        #crosshair {{ width: 8px; height: 8px; border: 2px solid white;
                      border-radius: 50%; box-shadow: 0 0 4px rgba(0,0,0,0.5); }}
        #hint {{ position: absolute; bottom: 16px; left: 50%;
                 transform: translateX(-50%);
                 color: rgba(255,255,255,0.7); font-size: 14px;
                 font-family: monospace; pointer-events: none; 
                 background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 5px; }}
        #room-label {{ position: absolute; top: 16px; left: 16px;
                       color: white; font-size: 15px; font-family: monospace;
                       background: rgba(0,0,0,0.5); padding: 5px 10px;
                       border-radius: 4px; pointer-events: none; }}
        #fullscreenBtn {{ position: absolute; top: 16px; right: 220px;
                         background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.3);
                         color: white; padding: 6px 12px; border-radius: 4px;
                         cursor: pointer; font-size: 14px; font-family: monospace; z-index: 10; }}
        #fullscreenBtn:hover {{ background: rgba(255,255,255,0.15); }}
    </style>
    
    <div id="container">
        <canvas id="renderCanvas"></canvas>
        <canvas id="minimap" width="200" height="200"></canvas>
        <div id="hud"><div id="crosshair"></div></div>
        <div id="hint">Right-click drag to look \u00b7 WASD to move \u00b7 Left-click doors to open</div>
        <div id="room-label">Interactive BIM Navigator</div>
        <button id="fullscreenBtn" onclick="toggleFS()">\u26F6 Fullscreen</button>
    </div>
    
    <script>
        const container = document.getElementById("container");
        const canvas = document.getElementById("renderCanvas");
        const engine = new BABYLON.Engine(canvas, true);
        const scene = new BABYLON.Scene(engine);
        
        scene.useRightHandedSystem = true; 
        scene.clearColor = new BABYLON.Color4(0.08, 0.08, 0.18, 1.0);
        
        const START_X = {start_x};
        const START_Z = {start_z};
        const GLB_B64 = "{glb_b64}";
        const DOOR_META = {door_json};
        const WALL_VECTORS = {walls_json};
        
        // ========== Fullscreen Toggle ==========
        function toggleFS() {{
            if (!document.fullscreenElement) {{
                container.requestFullscreen().catch(err => {{}});
            }} else {{
                document.exitFullscreen();
            }}
        }}
        document.addEventListener("fullscreenchange", () => {{ engine.resize(); }});
        
        // ========== Camera ==========
        const camera = new BABYLON.UniversalCamera("fps", new BABYLON.Vector3(START_X, 1.6, START_Z), scene);
        camera.setTarget(new BABYLON.Vector3(START_X, 1.6, START_Z + 1));
        camera.attachControl(canvas, true);
        
        camera.keysUp = [87];    // W
        camera.keysDown = [83];  // S
        camera.keysLeft = [65];  // A
        camera.keysRight = [68]; // D
        camera.speed = 0.15;
        camera.angularSensibility = 2000;
        
        camera.checkCollisions = true;
        camera.applyGravity = true;
        camera.ellipsoid = new BABYLON.Vector3(0.3, 0.8, 0.3);
        scene.gravity = new BABYLON.Vector3(0, -9.81, 0);
        
        // ========== Mouse Look via Right-Click Drag ==========
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;
        const MOUSE_SENS = 0.003;
        
        canvas.addEventListener("mousedown", (e) => {{
            if (e.button === 2) {{  // Right-click to drag
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                e.preventDefault();
            }}
        }});
        canvas.addEventListener("mouseup", (e) => {{
            if (e.button === 2) isDragging = false;
        }});
        canvas.addEventListener("mousemove", (e) => {{
            if (isDragging) {{
                const dx = e.clientX - lastMouseX;
                const dy = e.clientY - lastMouseY;
                camera.rotation.y += dx * MOUSE_SENS;
                camera.rotation.x += dy * MOUSE_SENS;
                camera.rotation.x = Math.max(-Math.PI/2.5, Math.min(Math.PI/2.5, camera.rotation.x));
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            }}
        }});
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        
        // ========== Lights ==========
        const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
        light.intensity = 1.0;
        const light2 = new BABYLON.HemisphericLight("light2", new BABYLON.Vector3(0, -1, 0), scene);
        light2.intensity = 0.3;
        
        // ========== Load GLB ==========
        BABYLON.SceneLoader.ImportMesh(
            "", "", "data:model/gltf-binary;base64," + GLB_B64,
            scene,
            (meshes) => {{
                meshes.forEach(mesh => {{
                    mesh.checkCollisions = true;
                }});
            }}
        );
        
        // ========== Door System with Hinge Pivot ==========
        const doors = {{}};
        DOOR_META.forEach(door => {{
            const bx = door.center_m[0];
            const by = door.height_m / 2.0;
            const bz = door.center_m[1];
            const wallAngle = door.wall_angle_rad;
            const halfW = door.width_m / 2.0;
            
            // 1. Create a pivot TransformNode at the HINGE EDGE of the door
            const pivot = new BABYLON.TransformNode(door.id + "_pivot", scene);
            // Hinge position = door center offset by half-width along the wall direction
            const hingeX = bx - Math.cos(wallAngle) * halfW;
            const hingeZ = bz + Math.sin(wallAngle) * halfW;
            pivot.position = new BABYLON.Vector3(hingeX, 0, hingeZ);
            pivot.rotation.y = -wallAngle;
            
            // 2. Create the door panel, offset from pivot so it swings from the edge
            const panel = BABYLON.MeshBuilder.CreateBox(door.id, {{
                width: door.width_m,
                height: door.height_m,
                depth: 0.08
            }}, scene);
            
            const mat = new BABYLON.StandardMaterial(door.id + "_mat", scene);
            mat.diffuseColor = new BABYLON.Color3(0.45, 0.25, 0.10);
            mat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
            panel.material = mat;
            
            // Offset panel so its edge aligns with the pivot point (the hinge)
            panel.parent = pivot;
            panel.position = new BABYLON.Vector3(halfW, by, 0);
            panel.checkCollisions = true;
            
            doors[door.id] = {{
                pivot: pivot,
                mesh: panel,
                isOpen: false,
                baseAngle: pivot.rotation.y
            }};
        }});
        
        // ========== Door Interaction — Left-Click Raycast ==========
        scene.onPointerDown = (evt) => {{
            if (evt.button !== 0) return;  // Left-click only
            const ray = scene.createPickingRay(
                engine.getRenderWidth() / 2,
                engine.getRenderHeight() / 2,
                BABYLON.Matrix.Identity(),
                camera
            );
            const hit = scene.pickWithRay(ray);
            if (hit.pickedMesh) {{
                // Check if the picked mesh is a door panel
                const doorId = hit.pickedMesh.name;  
                if (doors[doorId]) {{
                    const door = doors[doorId];
                    const dist = BABYLON.Vector3.Distance(camera.position, hit.pickedPoint);
                    if (dist < 3.0) {{  // 3m interaction radius
                        const targetAngle = door.isOpen ? door.baseAngle : door.baseAngle + Math.PI / 2;
                        BABYLON.Animation.CreateAndStartAnimation(
                            "doorSwing", door.pivot, "rotation.y",
                            60, 24,
                            door.pivot.rotation.y, targetAngle,
                            BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
                        );
                        door.isOpen = !door.isOpen;
                        door.mesh.checkCollisions = !door.isOpen;
                    }}
                }}
            }}
        }};
        
        // ========== Minimap ==========
        const mmCanvas = document.getElementById("minimap");
        const mmCtx = mmCanvas.getContext("2d");
        const MM_SIZE = 200;
        const MM_PAD = 10;
        
        let minX = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxZ = -Infinity;
        WALL_VECTORS.forEach(pair => {{
            pair.forEach(pt => {{
                minX = Math.min(minX, pt[0]); minZ = Math.min(minZ, pt[1]);
                maxX = Math.max(maxX, pt[0]); maxZ = Math.max(maxZ, pt[1]);
            }});
        }});
        
        const mmScaleX = (MM_SIZE - MM_PAD*2) / (maxX - minX || 1);
        const mmScaleZ = (MM_SIZE - MM_PAD*2) / (maxZ - minZ || 1);
        const mmScale = Math.min(mmScaleX, mmScaleZ);
        
        function worldToMinimap(wx, wz) {{
            return {{
                x: MM_PAD + (wx - minX) * mmScale,
                y: MM_PAD + (wz - minZ) * mmScale
            }};
        }}
        
        const wallCache = document.createElement("canvas");
        wallCache.width = wallCache.height = MM_SIZE;
        const wcCtx = wallCache.getContext("2d");
        wcCtx.fillStyle = "rgba(20, 20, 40, 0.85)";
        wcCtx.fillRect(0, 0, MM_SIZE, MM_SIZE);
        wcCtx.strokeStyle = "rgba(255,255,255,0.8)";
        wcCtx.lineWidth = 2.0;
        WALL_VECTORS.forEach(pair => {{
            const p1 = worldToMinimap(pair[0][0], pair[0][1]);
            const p2 = worldToMinimap(pair[1][0], pair[1][1]);
            wcCtx.beginPath(); wcCtx.moveTo(p1.x, p1.y); wcCtx.lineTo(p2.x, p2.y); wcCtx.stroke();
        }});
        wcCtx.fillStyle = "#FFA500";
        Object.values(doors).forEach(door => {{
            const pos = worldToMinimap(door.pivot.position.x, door.pivot.position.z);
            wcCtx.fillRect(pos.x - 3, pos.y - 3, 6, 6);
        }});
        
        // ========== Render Loop ==========
        engine.runRenderLoop(() => {{
            scene.render();
            mmCtx.clearRect(0, 0, MM_SIZE, MM_SIZE);
            mmCtx.drawImage(wallCache, 0, 0);
            
            const pPos = worldToMinimap(camera.position.x, camera.position.z);
            mmCtx.fillStyle = "#4FC3F7";
            mmCtx.beginPath(); mmCtx.arc(pPos.x, pPos.y, 4, 0, Math.PI * 2); mmCtx.fill();
            
            const fwd = camera.getForwardRay().direction;
            mmCtx.strokeStyle = "#4FC3F7";
            mmCtx.lineWidth = 2.0;
            mmCtx.beginPath(); mmCtx.moveTo(pPos.x, pPos.y); 
            mmCtx.lineTo(pPos.x + fwd.x * 12, pPos.y + fwd.z * 12); mmCtx.stroke();
        }});
        
        window.addEventListener("resize", () => engine.resize());
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=height)

# --- HELPERS ---
def draw_vectors_on_image(shape, vectors, color=(0, 255, 0), thickness=2):
    canvas = np.zeros((*shape[:2], 3), dtype=np.uint8)
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
    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_file_id") != file_id:
            # New file uploaded — clear previous results
            for key in ["glb_bytes", "door_meta", "vectors",
                        "demo_glb_post", "demo_glb_pre", "demo_door_meta", "demo_vectors"]:
                st.session_state.pop(key, None)
            st.session_state["last_file_id"] = file_id
            

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
            # Viewer toggle lives OUTSIDE the run_btn block
            viewer_mode = st.radio(
                "Viewer Mode",
                ["🔭 Passive (model-viewer)", "🎮 Navigate (First-Person)"],
                horizontal=True
            )

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
                vectors, detections, door_cuts = pipeline.process_geometry( mask, original_image=bgr, yolo_weights=yolo_w)

            # 3D
            # 3D
            with st.spinner("Building interactive 3D model..."):
                # Generate Passive Mesh (Static doors baked in)
                mesh_passive, door_meta = pipeline.generate_3d_scene(vectors, detections, door_cuts, bake_doors=True)
                # Generate Navigation Mesh (No static doors, leaves gaps for JS)
                mesh_nav, _ = pipeline.generate_3d_scene(vectors, detections, door_cuts, bake_doors=False)
                
                if mesh_passive and mesh_nav:
                    # Store both versions in session state
                    st.session_state["glb_bytes_passive"] = mesh_to_glb_bytes(mesh_passive)
                    st.session_state["glb_bytes_nav"] = mesh_to_glb_bytes(mesh_nav)
                    st.session_state["door_meta"] = door_meta
                    st.session_state["vectors"] = vectors
                    
                    with col2:
                        det_msg = f" · {len(detections)} icons detected" if detections else ""
                        st.success(f"✅ {len(vectors)} walls reconstructed{det_msg}")
                else:
                    st.error("No geometry detected.")

        if "glb_bytes_passive" in st.session_state:
            with col2:
                det_msg = f" · {len(st.session_state.get('door_meta', []))} doors" if st.session_state.get('door_meta') else ""
                st.success(f"✅ Model ready{det_msg}")
                
                # Dynamically load the correct GLB based on the viewer selected
                if viewer_mode.startswith("🔭"):
                    render_3d_viewer(st.session_state["glb_bytes_passive"])
                else:
                    render_navigation_viewer(
                        st.session_state["glb_bytes_nav"],
                        st.session_state.get("door_meta", []),
                        st.session_state.get("vectors", [])
                    )
                st.download_button("📥 Download GLB", st.session_state["glb_bytes_passive"],
                                "floorplan.glb", "model/gltf-binary")
    # ============================
    #  DEMONSTRATIVE MODE
    # ============================
    else:
        st.image(image_rgb, caption=f"Input ({image.shape[1]}×{image.shape[0]})", width="stretch")
        viewer_mode = st.radio(
            "Viewer Mode",
            ["🔭 Passive (model-viewer)", "🎮 Navigate (First-Person)"],
            horizontal=True
        )
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
            all_tracked_dets = []
            door_cuts = []

            if yolo_available:
                with st.status("Stage 4: YOLOv8 Detection & Structural Correction", expanded=True) as status:
                    is_fast = (inference_mode == "Fast (512px)")
                    vectors_corrected, all_tracked_dets, door_cuts = structural_corrector.correct_structure(image, vectors_5b, is_fast_mode=is_fast)
                    
                    discarded_doors = [d for d in all_tracked_dets if d.get("status") == "discarded" and d["class"] in structural_corrector.DOOR_CLASSES]
                    valid_doors = [d for d in all_tracked_dets if d.get("status") == "valid_door"]
                    furniture_dets = [d for d in all_tracked_dets if d["class"] in pipeline.FURNITURE_ASSETS]

                    c1_yolo, c2_yolo, c3_yolo = st.columns(3)
                    with c1_yolo:
                        discard_vis = draw_yolo_boxes(image_rgb, discarded_doors)
                        st.image(discard_vis, caption=f"Discarded Doors ({len(discarded_doors)})", width="stretch")
                    with c2_yolo:
                        valid_vis = draw_yolo_boxes(image_rgb, valid_doors)
                        st.image(valid_vis, caption=f"Used Doors ({len(valid_doors)})", width="stretch")
                    with c3_yolo:
                        furn_vis = draw_yolo_boxes(image_rgb, furniture_dets)
                        st.image(furn_vis, caption=f"Furniture ({len(furniture_dets)})", width="stretch")

                    vectors_final = vectors_corrected

                    c1, c2 = st.columns(2)
                    with c1:
                        pre_corr_img = draw_vectors_on_image(image.shape, vectors_5b, color=(0, 255, 0))
                        st.image(cv2.cvtColor(pre_corr_img, cv2.COLOR_BGR2RGB),
                                 caption=f"Before Correction ({len(vectors_5b)})", width="stretch")
                    with c2:
                        post_corr_img = draw_vectors_on_image(image.shape, vectors_corrected, color=(0, 255, 128))
                        post_corr_img = draw_doors_on_vector_map(post_corr_img, all_tracked_dets)
                        st.image(cv2.cvtColor(post_corr_img, cv2.COLOR_BGR2RGB),
                                 caption=f"After Correction ({len(vectors_corrected)})", width="stretch")

                    status.update(label="Stage 4: Complete ✅", state="complete")
            else:
                st.warning("⚠️ best.pt not found — Skipping Phase 2C structural correction")

            # ── Stage 5: 3D Model Generation ──
            with st.status("Stage 5: 3D Model Construction", expanded=True) as status:
                mesh_pre, _ = pipeline.generate_3d_scene(vectors_5b)
                # Passive mode: bake doors into the GLB mesh
                mesh_post_passive, door_meta = pipeline.generate_3d_scene(vectors_final, all_tracked_dets, door_cuts, bake_doors=True)
                # Navigation mode: doors are spawned by Babylon.js, so DON'T bake them into the GLB
                mesh_post_nav, _ = pipeline.generate_3d_scene(vectors_final, all_tracked_dets, door_cuts, bake_doors=False)
                
                viewer_mode = st.radio(
                    "Viewer Mode",
                    ["🔭 Passive (model-viewer)", "🎮 Navigate (First-Person)"],
                    horizontal=True,
                    key="viewer_mode_stage5"
                )
                
                is_nav = viewer_mode.startswith("🎮")
                
                if mesh_pre and mesh_post_passive and yolo_available:
                    glb_pre = mesh_to_glb_bytes(mesh_pre)
                    glb_post_passive = mesh_to_glb_bytes(mesh_post_passive)
                    glb_post_nav = mesh_to_glb_bytes(mesh_post_nav) if mesh_post_nav else glb_post_passive

                    st.subheader("Pre-Correction Model")
                    if not is_nav:
                        render_3d_viewer(glb_pre, height=450)
                    else:
                        render_navigation_viewer(glb_pre, [], vectors_5b, height=450)
                        
                    st.download_button("📥 Download Pre-Correction GLB", glb_pre,
                                       "floorplan_pre.glb", "model/gltf-binary", key="dl_pre")

                    st.subheader("Post-Correction Model")
                    if not is_nav:
                        render_3d_viewer(glb_post_passive, height=450)
                    else:
                        render_navigation_viewer(glb_post_nav, door_meta, vectors_final, height=700)
                        
                    st.download_button("📥 Download Post-Correction GLB", glb_post_passive,
                                       "floorplan_post.glb", "model/gltf-binary", key="dl_post")
                elif mesh_post_passive:
                    glb_post_passive = mesh_to_glb_bytes(mesh_post_passive)
                    glb_post_nav = mesh_to_glb_bytes(mesh_post_nav) if mesh_post_nav else glb_post_passive
                    if not is_nav:
                        render_3d_viewer(glb_post_passive)
                    else:
                        render_navigation_viewer(glb_post_nav, door_meta, vectors_final, height=700)
                    st.download_button("📥 Download GLB", glb_post_passive,
                                       "floorplan.glb", "model/gltf-binary")
                else:
                    st.error("No geometry detected.")

                status.update(label="Stage 5: Complete ✅", state="complete")
                if mesh_post_nav:
                    st.session_state["demo_glb_nav"] = mesh_to_glb_bytes(mesh_post_nav)
                if mesh_post_passive:
                    st.session_state["demo_glb_post"] = mesh_to_glb_bytes(mesh_post_passive)
                    st.session_state["demo_door_meta"] = door_meta
                    st.session_state["demo_vectors"] = vectors_final
                if mesh_pre:
                    st.session_state["demo_glb_pre"] = mesh_to_glb_bytes(mesh_pre)

            # ── Summary ──
            st.divider()
            st.success(
                f"**Pipeline Complete!** "
                f"{len(final_raw)} raw → {len(vectors_5b)} post-5B → {len(vectors_final)} final vectors"
                + (f" · {len(detections)} YOLO detections used" if detections else "")
            )
        
        if "demo_glb_post" in st.session_state:
            viewer_mode = st.radio(
                "Viewer Mode",
                ["🔭 Passive (model-viewer)", "🎮 Navigate (First-Person)"],
                horizontal=True,
                key="demo_viewer_mode_after_run"
            )
            st.subheader("Post-Correction Model")
            if viewer_mode.startswith("🔭"):
                render_3d_viewer(st.session_state["demo_glb_post"], height=450)
            else:
                render_navigation_viewer(
                    st.session_state.get("demo_glb_nav", st.session_state["demo_glb_post"]),
                    st.session_state.get("demo_door_meta", []),
                    st.session_state.get("demo_vectors", []),
                    height=700
                )


if __name__ == "__main__":
    main()