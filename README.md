# 🏗️ Scan-to-BIM

**AI-Driven Floorplan → BIM-Grade 3D Reconstruction**

This project implements a **research-grade Scan-to-BIM pipeline** that converts a **2D architectural floorplan** into a **metric-scaled, editable 3D building model (GLB)** using deep learning, computational geometry, and raster-to-vector conversion.

Unlike mesh generators, this system reconstructs **architectural geometry**:
walls, doors, topology, and scale.

---

## 🔥 What This Project Does

**Input**
A raster floorplan image (PNG / JPG)

**Output**
A **true-scale BIM-style 3D model** with:

* Walls as solids
* Door openings with headers (lintels)
* Floor slab
* Correct metric dimensions
* Exportable `.glb`
* Interactive browser viewer

This is a **geometry reconstruction engine**, not a visualizer.

---

## 🧠 System Architecture

```
Floorplan Image
      │
      ▼
[ Phase 1 ]  CNN → Wall Probability Map
      │
      ▼
[ Phase 4 ]  Tiled High-Res Mask Generation
      │
      ▼
[ Phase 5 ]  Raster → Vector Geometry
      │
      ▼
[ Phase 3 ]  Procedural BIM Construction
      │
      ▼
     GLB 3D Building
```

Each phase is modular and independently upgradable.

---

# 🧩 Phase Breakdown

---

## **Phase 1 — Perception (Deep Learning)**

**Status: Implemented**

* Model: **UNet (ResNet-34 encoder)**
* Task: Pixel-wise wall probability estimation
* Output: Floating-point wall probability map

Features:

* ImageNet-normalized inference
* Aspect-ratio preserved resizing
* GPU-accelerated PyTorch pipeline

---

## **Phase 4A — High-Resolution Tiled Inference (SAHI-Style)**

**Status: Implemented**

Solves the “small-CNN vs large-floorplan” problem.

* Input images are split into **overlapping 512×512 tiles**
* Each tile is processed by the CNN
* Outputs are merged using **Hann-window weighted blending**

This preserves:

* Wall continuity across tile borders
* Large building geometry
* High-resolution detail

Inspired by **SAHI (Slicing Aided Hyper Inference)**.

---

## **Phase 4B — Mask Refinement (Noise & Gap Cleanup)**

**Status: Implemented**

The raw CNN probability map is cleaned using:

* Hard thresholding
* Connected-component filtering (dust removal)
* Morphological closing (gap bridging)
* Morphological opening (edge smoothing)

Output:

> A clean, contiguous wall mask suitable for vectorization

---

## **Phase 5A — Hybrid Raster-to-Vector Conversion**

**Status: Implemented**

This is the core **Scan-to-BIM** step.

It combines:

* **Skeleton topology** → connectivity
* **Pixel clouds** → geometric accuracy

Pipeline:

1. Skeletonize refined mask
2. Convert skeleton to graph
3. Trace wall paths
4. Split at corners using RDP
5. Extract wall pixel regions around each segment
6. Fit **least-squares PCA lines** to those pixels
7. Generate **CAD-grade wall axes**

This produces:

> Straight, metric-accurate, topology-aware wall vectors

---

## **Phase 5B — Junction & Topology Optimization**

**Status: To be built**

Will implement:

* Vertex snapping
* Corner closure
* Manhattan-world (90°) enforcement
* Room closure

Purpose:

> Convert straight lines into a **topologically valid floorplan**

---

## **Phase 3 — BIM-Style 3D Construction**

**Status: Implemented**

Using **Trimesh**, vectors are converted into solids:

* Pixel → meter scaling
* Wall extrusion
* Door gap detection
* Header (lintel) generation
* Floor slab
* Watertight GLB mesh

---

## **Phase 6 — Parametric BIM**

**Status: To be built**

* Editable wall thickness
* Door sizes
* IFC / Revit export

---

## **Phase 7 — Multi-Floor & Scale Calibration**

**Status: To be built**

* Stair detection
* Floor stacking
* Absolute scale calibration

---

# 🖥️ Web Interface

Streamlit-based UI:

* Upload a floorplan
* Choose inference mode (Fast / High-Fidelity)
* Run full Scan-to-BIM pipeline
* View 3D model in browser
* Download GLB

Viewer uses **Google `<model-viewer>`** embedded via Base64.

---

# 📦 Repository

```
app.py        → Web UI
pipeline.py   → Full Scan-to-BIM engine
model_links.txt
sample_io/
```

---

# ✅ Current Capabilities

| Feature                      | Status |
| ---------------------------- | ------ |
| CNN wall detection           | ✅      |
| Tiled inference (SAHI-style) | ✅      |
| Mask cleanup                 | ✅      |
| Skeleton topology            | ✅      |
| Pixel-cloud line fitting     | ✅      |
| CAD-grade wall vectors       | ✅      |
| Door detection               | ✅      |
| Headers (lintels)            | ✅      |
| Metric scaling               | ✅      |
| 3D BIM model                 | ✅      |
| Web viewer                   | ✅      |

---

# ⚠️ Known Limitations

* Some thin walls may be missed (CNN)
* Corners may not perfectly close (Phase-5B pending)
* Windows & room semantics not yet modeled
* Single-floor only

These are **expected** and solved in upcoming phases.

---

# 🏆 Why This Project Is Different

Most systems output **meshes**.
This system outputs **architecture**.

It models:

* topology
* geometry
* scale
* doors
* walls
* floors

This is what makes it **BIM-grade**.

---

# 🚀 Status

**V1.5 — High-Resolution, Geometry-Accurate Scan-to-BIM Engine**

The system now produces **CAD-grade straight walls** from raw images.
Next phases will enforce **topological correctness and semantics**.


