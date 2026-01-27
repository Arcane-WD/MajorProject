import streamlit as st
import numpy as np
import cv2
import torch
import tempfile
import base64
import pipeline  # Importing our logic file

# --- CONFIG ---
st.set_page_config(page_title="Scan-to-BIM Engine", page_icon="🏗️", layout="wide")

# --- CACHED MODEL LOADING ---
@st.cache_resource
def get_model():
    """Wrapper to cache the model so we don't reload it on every interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path to your model file
    MODEL_PATH = "best_cleaner_model.pth"
    try:
        model = pipeline.load_model_logic(MODEL_PATH, device)
        return model, device
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None

# --- VIEWER COMPONENT ---
def render_3d_viewer(glb_bytes):
    b64 = base64.b64encode(glb_bytes).decode('utf-8')
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <style>
          model-viewer {{
            width: 100%;
            height: 600px;
            background-color: #f0f2f6;
            border-radius: 10px;
          }}
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
          max-camera-orbit="infinitydeg 180deg auto"
        ></model-viewer>
      </body>
    </html>
    """
    st.components.v1.html(html_code, height=600)

# --- MAIN APP ---
def main():
    st.title("🏗️ Scan-to-BIM: AI Floorplan Reconstructor")
    
    # Load Model
    model, device = get_model()
    if not model:
        st.warning("Please move 'best_cleaner_model.pth' into this folder.")
        st.stop()
        
    st.sidebar.success(f"Engine Online ({device})")
    
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns([1, 2])

    if uploaded_file:
        # Read Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(image, caption="Input Floorplan", width = 'stretch')
            run_btn = st.button("Generate 3D Model", type="primary")

        if run_btn:
            with st.spinner("1/3 Perception..."):
                mask = pipeline.predict_mask(model, device, image)
            
            with st.spinner("2/3 Geometry..."):
                vectors = pipeline.process_geometry(mask)
            
            with st.spinner("3/3 Construction..."):
                mesh = pipeline.generate_3d_scene(vectors)
            
            if mesh:
                # Export to temp GLB
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                    mesh.export(tmp.name)
                    with open(tmp.name, "rb") as f:
                        glb_bytes = f.read()
                
                with col2:
                    st.success(f"Built {len(vectors)} wall segments.")
                    render_3d_viewer(glb_bytes)
                    st.download_button("Download GLB", glb_bytes, "floorplan.glb", "model/gltf-binary")
            else:
                st.error("No geometry detected. Try a cleaner image.")

if __name__ == "__main__":
    main()