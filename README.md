# Scan-to-BIM: 2D Floorplan to 3D Model Pipeline

An end-to-end system that converts a 2D floorplan image into a structured 3D architectural model using a hybrid **Deep Learning + Computer Vision + Geometry** pipeline.

This project focuses on **system reliability first**, followed by visual fidelity — making it suitable for real-world Scan-to-BIM style applications.

---

## 🚀 What This Project Does

**Input:**  
A raster floorplan image (with furniture, text, noise)

**Output:**  
A valid `.glb` 3D model containing:
- Scaled walls  
- Door openings  
- Headers (lintels)  
- Floor slab  

---

## 🧠 System Architecture

```

Phase 1 — Perception (Deep Learning)
↓
Phase 2 — Drafting (Computer Vision + Graph Theory)
↓
Phase 3 — Construction (3D Geometry)
↓
GLB Export

```

---

## 🔹 Phase Breakdown

### Phase 1 — Perception
- U-Net (ResNet34 backbone)
- Binary semantic segmentation: wall vs non-wall
- Trained on preprocessed CubiCasa5K images
- Output: clean wall mask

### Phase 2 — Drafting
- Morphological closing to repair gaps  
- Skeletonization to get wall centerlines  
- Graph construction (4-connectivity)  
- RDP simplification  
- Produces clean vector wall segments

### Phase 3 — Construction
- Metric scaling (pixel → meters)
- Robust wall generation using rotated 3D boxes
- Automatic header (lintel) detection above doors
- Floor slab generation
- GLB export via `trimesh`

---

## 📂 Repository Structure

```

.
├── pipeline_v1.py          # Full end-to-end pipeline (V1.1)
├── model_links.txt        # Drive links for trained model + dataset
└── README.md

````

---

## 📦 Assets

Due to size constraints, large assets are hosted externally:

- **Trained Model:** `best_cleaner_model.pth`  
- **Preprocessed Dataset:** `processed_dataset_512.zip`  

Links are provided in:  
`model_links.txt`

---

## 🏗️ Current Status

The system currently supports:

- Single-image inference  
- Batch processing  
- Automatic geometry generation  
- Exportable 3D models  

The focus so far has been:
> **Correct geometry first, visual fidelity second**

---

## 🗺️ Roadmap

Next phases will introduce:

- High-resolution tiled inference  
- Robust door/window classification  
- Room detection  
- Boolean mesh unions  
- Web-based GLTF viewer integration  
- CAD export (DXF/IFC)

---

## ⚙️ Requirements

```bash
pip install segmentation-models-pytorch
pip install trimesh networkx rdp scikit-image
pip install opencv-python matplotlib
````

---

## 👤 Author

Building as a full-stack ML + geometry systems project to demonstrate:

* Applied deep learning
* Computer vision pipelines
* Graph-based geometry reasoning
* 3D procedural modeling

````

---

# ✅ Project Progress Checklist

You asked for a **checkmark table** — here’s a clean one you can add under a section called `## Progress`.

```markdown
## ✅ Project Progress

| Phase | Component | Status |
|------:|-----------|:------:|
| 1 | Dataset preprocessing | ✔️ |
| 1 | Wall segmentation model trained | ✔️ |
| 1 | Inference pipeline | ✔️ |
| 2 | Morphological cleanup | ✔️ |
| 2 | Skeletonization | ✔️ |
| 2 | Graph-based vectorization | ✔️ |
| 2 | RDP simplification | ✔️ |
| 3 | Metric scaling | ✔️ |
| 3 | Wall extrusion | ✔️ |
| 3 | Floor slab generation | ✔️ |
| 3 | Door gap detection | ✔️ |
| 3 | Header (lintel) logic | ✔️ |
| 3 | GLB export | ✔️ |
| 4 | Tiled high-res inference | ⬜ |
| 4 | Vector accuracy refinement | ⬜ |
| 4 | Robust door/window classification | ⬜ |
| 5 | Room detection | ⬜ |
| 5 | Boolean mesh unions | ⬜ |
| 6 | Web-based GLTF viewer | ⬜ |
| 6 | Real-time interaction UI | ⬜ |
| 7 | CAD export (DXF/IFC) | ⬜ |
````
