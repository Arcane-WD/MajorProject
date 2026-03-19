# Topic 5: 3D Asset Placement and Canonical Scaling

After extracting a pristine 2D vector map of the architecture and structurally slicing physical gaps into the wall vectors mathematically, we must convert this lightweight topological array into a tangible 3D Building Information Model (BIM).

---

## 1. Trimesh Extrusions (Walls & Floors)

- **What is Trimesh?** A pure Python library for loading and using triangular meshes.
- **The Process:** A 3D architectural wall is fundamentally a very simple rectangular prism constructed along a vector path.
  1. We iterate through every topological vector $(p_1, p_2)$ in our final mapped array.
  2. We multiply the pixel coordinates by `PIXEL_TO_METER = 0.05` to transform the space from a 2D image coordinate system into a 1:1 real-world physical metric simulation.
  3. We create a parametric boundary box at the origin measuring `Length` (distance from $p_1$ to $p_2$), `Width` (`WALL_THICKNESS`), and `Height` (`WALL_HEIGHT`).
  4. We physically translate (`.apply_translation()`) and rotate (`.apply_transform(rotation_matrix)`) the 3D solid off the origin grid so it perfectly bounds the underlying 2D vector points!

---

## 2. `.glb` Asset Placements & Transformations

For complex, organic architectural components like Beds, Sofas, and Chairs, primitive cube math is insufficient. Instead, we dynamically inject predefined `.glb` (GL Transmission Format) 3D meshes into our scene.

### Coordinate Space Rotation (Z-Up vs Y-Up)
A classic pitfall in 3D programming is coordinate space alignment. 
- **Y-up Systems:** Most 3D modeling software (Blender, Unity) define the Y-axis as the ceiling.
- **Z-up Systems:** Our architectural environment plots XY flatly as the floor grid (like a traditional blueprint), rendering Z as the height ceiling.

If we blindly import a `.glb` bed, it natively believes Y is "Up", and consequently will spawn lying on its side.
**Solution:**
```python
# Force Y-up 3D meshes into global Z-up Space
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
```
We aggressively multiply the asset against a $90^\circ$ X-axis rotational matrix exactly once upon mesh generation, normalizing its spatial axes to match the floorplan.

---

## 3. Scale Constraints: Canonical vs Dynamic Bounding

How big should a 3D table be?

### The Dynamic Bounding Box Flaw
We originally coded a system where the `.glb` furniture asset stretched dynamically to literally fill the exact width and height of the YOLO detection coordinates natively drawn on the screen.
- **The Core Issue:** A YOLO bounding box labeled "Dining Table" inherently encompasses the table *and all the chairs surrounding it*. 
- Forcing a single `.glb` Table geometry to scale up to fit the area of a Table *plus* 6 chairs results in a mathematically gargantuan, distorted table taking up the entire room.

### Strategy: Canonical Target Scaling
To solve geometric bloat, we reverted to **Strict Canonical Constraints**, identical to the Door Carving logic.
1. We predefine literal physical constraints for every asset class inside a dictionary (e.g., `(2.0, 1.4)` meters for a double bed).
2. We extract the native object bounding extents (`mesh.extents`). Since the object was rotated, `extents[0]` is Width, and `extents[2]` is Depth.
3. We generate a scale matrix to proportionately stretch the native un-scaled mesh until it explicitly aligns with our mathematically predefined constraint targets.

```python
extents = mesh.extents
scale_x = length_m / (extents[0] if extents[0] > 0 else 1)
scale_z = width_m / (extents[2] if extents[2] > 0 else 1)

scale_matrix = np.eye(4)
scale_matrix[0,0], scale_matrix[1,1], scale_matrix[2,2] = scale_x, min(scale_x, scale_z), scale_z
mesh.apply_transform(scale_matrix)
```

By binding spatial assets to canonical metrics instead of dynamic inference masks, the scene preserves architectural proportion globally without compromising placement accuracy.

---

## External Resources & Further Reading
1. **Trimesh Documentation:** [Trimesh - Python 3D Library](https://trimesh.org/)
2. **GL Transmission Format:** [GLB (glTF Binary) Format](https://www.khronos.org/gltf/)
3. **Similarity Transforms (Translation, Rotation, Scale):** [3D Affine Transformations](https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations)
