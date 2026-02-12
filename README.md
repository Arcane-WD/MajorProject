# Scan-to-BIM

**AI-Driven 2D Floorplan to BIM-Grade 3D Reconstruction**

This project implements a **research-oriented Scan-to-BIM pipeline** that converts a **2D architectural floorplan image** into a **metric-scaled, editable 3D building model (GLB)** using deep learning, computational geometry, and raster-to-vector reconstruction.

Unlike mesh-based visualizers, the system reconstructs **architectural structure**—walls, doors, topology, and scale—suitable for BIM-style downstream use.

---

## Project Objective

### Input

A raster floorplan image (PNG / JPG)

### Output

A **metric-scale 3D building model** containing:

* Walls represented as solids
* Door openings with headers (lintels)
* Floor slab
* Consistent pixel-to-meter scaling
* Exportable `.glb` model
* Interactive browser-based visualization

The system focuses on **geometric reconstruction**, not surface visualization.

---

## System Architecture

```
Floorplan Image
      │
      ▼
[ Phase 1 ]  CNN-Based Wall Probability Estimation
      │
      ▼
[ Phase 4 ]  High-Resolution Mask Generation
      │
      ▼
[ Phase 5 ]  Raster-to-Vector Geometry Extraction
      │
      ▼
[ Phase 3 ]  Procedural BIM-Style Construction
      │
      ▼
     GLB 3D Building Model
```

Each phase is modular and independently extensible.

---

## Phase Overview

---

### Phase 1 — Perception (Deep Learning)

**Status:** Implemented

* Model: UNet with ResNet-34 encoder
* Task: Pixel-wise wall probability estimation
* Output: Floating-point probability map

Key characteristics:

* ImageNet-normalized inference
* Aspect-ratio–preserving resizing
* GPU-accelerated PyTorch pipeline

---

### Phase 4A — High-Resolution Tiled Inference

**Status:** Implemented

Addresses the resolution mismatch between fixed-input CNNs and large floorplans.

* Input images are divided into overlapping 512×512 tiles
* Each tile is independently processed by the CNN
* Outputs are merged using Hann-window weighted blending

This preserves wall continuity and global geometry while maintaining high spatial resolution.

This approach is conceptually aligned with SAHI-style inference.

---

### Phase 4B — Mask Refinement

**Status:** Implemented

Post-processing of CNN probability maps to ensure structural integrity before vectorization.

Operations include:

* Hard thresholding
* Connected-component filtering for noise removal
* Morphological closing to bridge small gaps
* Morphological opening for edge smoothing

The output is a clean, contiguous wall mask suitable for geometric processing.

---

### Phase 5A — Hybrid Raster-to-Vector Conversion

**Status:** Implemented

This phase performs the core Scan-to-BIM transformation by combining:

* Skeleton-based topology extraction
* Pixel-region–based geometric fitting

Pipeline steps:

1. Skeletonization of the refined mask
2. Conversion of the skeleton into a graph
3. Tracing of wall paths between junctions
4. Segmentation at geometric bends using RDP
5. Extraction of wall pixel regions around each segment
6. Least-squares (PCA) line fitting on pixel clouds
7. Generation of CAD-grade wall axes

This produces straight, metric-accurate wall vectors while preserving connectivity.

---

### Phase 5B — Junction and Topology Optimization

**Status:** Planned

Planned enhancements include:

* Vertex snapping and corner closure
* Manhattan-world (orthogonality) enforcement
* Room boundary closure

Objective: to produce a topologically consistent floorplan suitable for semantic reasoning.

---

### Phase 3 — BIM-Style 3D Construction

**Status:** Implemented

Using Trimesh, vector geometry is converted into 3D architectural solids:

* Pixel-to-meter scaling
* Wall extrusion with thickness and height
* Door gap detection
* Header (lintel) generation
* Floor slab construction
* Watertight GLB export

---

### Phase 6 — Parametric BIM Extensions

**Status:** Planned

* Editable wall thickness and door dimensions
* IFC / Revit-compatible export

---

### Phase 7 — Multi-Floor and Scale Calibration

**Status:** Planned

* Stair detection and floor stacking
* Absolute scale calibration

---

## Web Interface

A Streamlit-based interface provides:

* Floorplan upload
* Inference mode selection (Fast / High-Fidelity)
* End-to-end Scan-to-BIM execution
* Interactive 3D visualization
* GLB export

The viewer is implemented using Google’s `<model-viewer>` and Base64-embedded GLB rendering.

---

## Repository Structure

```
app.py        → Web interface
pipeline.py   → Scan-to-BIM processing pipeline
model_links.txt
sample_io/
```

---

## Current Capabilities

| Capability                      | Status      |
| ------------------------------- | ----------- |
| CNN-based wall detection        | Implemented |
| High-resolution tiled inference | Implemented |
| Mask refinement                 | Implemented |
| Skeleton-based topology         | Implemented |
| Pixel-cloud line fitting        | Implemented |
| CAD-grade wall vectors          | Implemented |
| Door detection                  | Implemented |
| Header (lintel) generation      | Implemented |
| Metric scaling                  | Implemented |
| BIM-style 3D model              | Implemented |
| Web-based visualization         | Implemented |

---

## Known Limitations

* Thin or ambiguous walls may be missed by the CNN
* Corner closure is not yet guaranteed (Phase 5B pending)
* Windows and room semantics are not yet modeled
* Single-floor support only

These limitations are expected and addressed in planned phases.

---

## Project Scope and Distinction

Most systems generate **surface meshes**.
This system reconstructs **architectural geometry**.

It explicitly models:

* Topology
* Geometry
* Scale
* Walls
* Doors
* Floors

This positions the project as a **BIM-oriented reconstruction pipeline**, rather than a visualization tool.

---

## Current Status

**Version 1.5 — High-Resolution, Geometry-Accurate Scan-to-BIM Engine**

The system currently produces straight, metric-consistent wall geometry from raster floorplans.
Upcoming phases will focus on topological correctness and semantic enrichment.
