# Topic 2: Vector Snapping and Manhattan Enforcement

After extracting raw mathematical vectors using the Ramer-Douglas-Peucker algorithm, the geometric graph is structurally incomplete. Micro-imperfections in the U-Net mask cause lines to slightly miss each other, creating walls that don't connect and rooms that aren't watertight.

---

## 1. Vector Snapping (Topological Closure)

Architectural CAD systems demand watertight geometries (where walls perfectly intersect to form closed polygonal rooms).

### The Geometric Problem
Two walls intended to form a $90^\circ$ corner might end at coordinates $(100.5, 50.1)$ and $(102.0, 49.8)$. If we attempt to extrude these into 3D, there will be a visible gap, and programmatic area-calculations for the room will fail.

### The Algorithm: Spatial KD-Tree Clustering
Instead of comparing every point to every other point ($O(N^2)$ time complexity), which is incredibly slow for large floorplans, we implement an optimized Euclidean distance check.
1. Extract all discrete endpoints from the vector array.
2. If the physical distance between two endpoints $p_a$ and $p_b$ is less than `SNAP_THRESHOLD` (e.g., $10\text{px}$):
   $$ \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} < \tau $$
3. We calculate the mathematical centroid (average) of the points and physically update the vector coordinates in the array so both segments share the exact same node coordinate.

### Code Walkthrough (`pipeline.py`)
```python
def close_gaps(vectors, snap_threshold=SNAP_THRESHOLD):
    # Iterates through lines and looks for floating endpoints
    # extending them to intersect with the closest perpendicular line
    # utilizing line-intersection formulas (y = mx + b derivations).
```

---

## 2. The Manhattan-World Assumption

Real-world architectural floorplans are overwhelmingly Cartesian. Walls are built at $0^\circ, 90^\circ, 180^\circ$, or $270^\circ$. 

### The Refinement Logic
Due to the rasterization staircase effect, a perfectly horizontal wall in a raw image might be extracted with a slight $1^\circ$ tilt.
We calculate the absolute angle of every extracted vector:
$$ \theta = \operatorname{arctan2}(y_2 - y_1, x_2 - x_1) $$

If $\theta$ falls within a tight tolerance $\epsilon$ of an orthogonal axis, we forcefully align it:
- **Horizontal Match:** If $\theta \approx 0$ or $\pi$, we calculate the average Y position: $Y_{avg} = (y_1 + y_2) / 2$. We then overwrite the vector as $((x_1, Y_{avg}), (x_2, Y_{avg}))$.
- **Vertical Match:** If $\theta \approx \pi/2$ or $3\pi/2$, we average the X coordinates identically.

```python
# From pipeline.py -> enforce_manhattan()
angle = np.arctan2(dy, dx)
if abs(angle) < ANGLE_TOLERANCE or abs(angle - np.pi) < ANGLE_TOLERANCE:
    # Force horizontal
    mean_y = (p1[1] + p2[1]) / 2.0
    vectors[i] = ((p1[0], mean_y), (p2[0], mean_y))
```

---

## 3. Healing Collinear Noise Gaps

Often, a physical wall in an image will have text overlapping it (like "BEDROOM"), causing the U-Net to predict a gap in the wall. This breaks one continuous wall into two separate wall vectors.

### Scenario A Detection
In `structural_corrector.py` -> `fill_small_gaps()`, we search for **Collinear Arrays**.
1. Compare all walls against all other walls.
2. If two walls are parallel (their dot product is $\approx 1$).
3. If the perpendicular distance between them is close to zero (they lie perfectly on the exact same infinite line).
4. If the gap between their closest endpoints is less than $15\text{px}$.
5. We merge the two vectors into a single unbroken segment spanning from the furthest endpoints.

*Note: We purposefully wait to carve doors until AFTER this step, to ensure the algorithm doesn't accidentally "heal" a real doorway.*

---

## External Resources & Further Reading
1. **The Manhattan World Assumption:** [Understanding the Manhattan World Assumption in Computer Vision](https://www.semanticscholar.org/paper/The-Manhattan-world-assumption%3A-regularities-in-Coughlan-Yuille/a906ff67de4bc08d2fc1baeb147391eaaf6fe48a)
2. **Euclidean Distances and Geometric Snapping:** [Spatial Indexing with KD-Trees](https://en.wikipedia.org/wiki/K-d_tree)
3. **Line Intersection Math:** [Line-Line Intersection](https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection)
