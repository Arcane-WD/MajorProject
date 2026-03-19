# Topic 5: 3D Asset Placement and Canonical Scaling

After extracting a 2D vector map of the architecture and structurally slicing the gaps for doors using YOLO metrics, we convert this mathematical topology into a tangible 3D Building Information Model (BIM).

## 1. Trimesh Extrusions (Walls/Floors)
- **Learning Point (Basics):** A 3D object can be built physically by drawing a very thin, flat shape (e.g., a 15cm wide rectangle along a wall path) and "extruding" it vertically to the ceiling height.
- **Learning Point (Advanced):** `Trimesh` handles 3D rendering mathematically using vectors. A solid wall is created natively using bounding widths `WALL_THICKNESS` and `WALL_HEIGHT` directly along our computed graph network lines. 
- **Code Reference:** `pipeline.py` -> `create_box()` takes a start coordinate ($p_1$), an end coordinate ($p_2$), mathematically identifies the orientation angle, builds a parametric rectangular block on the floor, and rotates it to match the wall angle perfectly.

## 2. `.glb` Asset Placements
For complex components like Furniture, we import external 3D meshes rather than modeling them recursively with math blocks.
- **Learning Point (Basics):** 3D artists build formats like `.glb` (GL Transmission Format) representing objects like "Beds" or "Sofas."
- **Coordinate Spaces (Z-up vs Y-up):** Many game-engine objects are designed natively "Y-up" (meaning the Y axis is "up" towards the ceiling). Our simulation is natively "Z-up" (XY acts as the floor grid, Z is height).
- **Code Reference:** `pipeline.py` -> When loading furniture via `load(...)`, we actively generate a matrix constraint `mesh.apply_transform(rotation_matrix(np.pi/2, [1, 0, 0]))`. This forces imported Y-up models strictly into Z-up space globally, physically aligning them correctly with the floor so they aren't "sleeping on their heads."

## 3. Scale Constraints (Canonical Footprints)
We need to stretch imported 3D representations to accurately display inside the 3D viewer.
- **Learning Point (Advanced):** YOLO bounding boxes represent the total visual dimension of clustered objects (a dining table bounding box inherently includes all the chairs). 
- **The Issue:** Attempting to force a single `.glb` table model to perfectly pack an entire bounding box results in mathematically inflated models (the table physically doubles in size to consume the chair region).
- **The Solution:** We implement strict **Canonical Width Extent Scaling**. We predefine the literal mathematical dimensions (e.g., `(2.0, 1.4)` for beds). We parse the native object's extents `.extents[0]` and `.extents[2]`, map them up directly to our canonical target physical space, and stretch them proportionately by multiplying the global object via `scale_matrix`.
- **Code Reference:** `pipeline.py`. Inside `FURNITURE_ASSETS`, we define constraints. `scale_x = length_m / extents[0]` correctly binds the mathematical width uniformly.

## External Resources & Further Reading
1. **Trimesh Documentation:** [Trimesh - Python 3D Library](https://trimesh.org/)
2. **GL Transmission Format:** [GLB (glTF Binary) Format](https://www.khronos.org/gltf/)
3. **Similarity Transforms (Translation, Rotation, Scale):** [3D Affine Transformations](https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations)
