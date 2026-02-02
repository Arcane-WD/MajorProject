import streamlit as st
import numpy as np
import cv2
import torch
import tempfile
import base64
import pipeline

# --- CONFIG ---
st.set_page_config(page_title="Scan-to-BIM Engine", page_icon="🏗️", layout="wide")

# --- CACHED MODEL ---
@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_cleaner_model.pth"
    try:
        model = pipeline.load_model_logic(MODEL_PATH, device)
        return model, device
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None

# --- VIEWER ---
def render_3d_viewer(glb_bytes):
    b64 = base64.b64encode(glb_bytes).decode('utf-8')
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <style>
          model-viewer {{ width: 100%; height: 600px; background-color: #f0f2f6; border-radius: 10px; }}
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
    st.components.v1.html(html_code, height=600)

# --- MAIN ---
def main():
    st.title("🏗️ Scan-to-BIM: AI Floorplan Reconstructor")
    
    # Sidebar
    st.sidebar.header("System Config")
    model, device = get_model()
    if not model: st.stop()
    st.sidebar.success(f"Engine Online ({device})")
    
    # Phase 4: Mode Selector
    mode = st.sidebar.radio("Inference Mode", ["Fast (512px)", "High Fidelity (Tiled)"])
    
    if mode == "Fast (512px)":
        st.sidebar.caption("⚠️ **Preview Only:** Non-metric scale. Useful for quick visual checks.")
    else:
        st.sidebar.caption("✅ **Metric Mode:** Preserves original scale and details.")

    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns([1, 2])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(image, caption=f"Input ({image.shape[1]}x{image.shape[0]})", width = 'stretch')
            run_btn = st.button("Generate 3D Model", type="primary")

        if run_btn:
            # 1. Perception
            if mode == "Fast (512px)":
                with st.spinner("Running Fast Perception..."):
                    mask = pipeline.predict_mask(model, device, image)
            else:
                progress_bar = st.progress(0, text="Running High-Fidelity Tiled Inference...")
                def update_progress(p):
                    progress_bar.progress(p, text=f"Stitching Tiles: {int(p*100)}%")
                
                mask = pipeline.predict_tiled(model, device, image, progress_callback=update_progress)
                progress_bar.empty()

            # 2. Geometry
            with st.spinner("Extracting Geometry..."):
                vectors = pipeline.process_geometry(mask)
            
            # 3. Construction
            with st.spinner("Constructing Building..."):
                mesh = pipeline.generate_3d_scene(vectors)
            
            if mesh:
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                    mesh.export(tmp.name)
                    with open(tmp.name, "rb") as f:
                        glb_bytes = f.read()
                
                with col2:
                    st.success(f"Success! {len(vectors)} walls reconstructed.")
                    render_3d_viewer(glb_bytes)
                    st.download_button("Download GLB", glb_bytes, "floorplan.glb", "model/gltf-binary")
            else:
                st.error("No geometry detected.")

if __name__ == "__main__":
    main()
    