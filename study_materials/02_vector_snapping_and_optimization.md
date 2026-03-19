# Topic 2: Vector Snapping and Manhattan Enforcement

Raster-to-vector extraction is inherently noisy. A floorplan straight wall might generate multiple micro-segments due to pixel staggering, and corners might not connect perfectly.

## 1. Vector Snapping (Closure)
- **Learning Point (Basics):** When two lines almost touch, but don't, we want to snap their endpoints together so the room creates a closed boundary (watertight).
- **Learning Point (Advanced):** We utilize a spatial clustering approach using a **KD-Tree** or pairwise distance matrices. If the Euclidean distance between two vector endpoints falls below a `SNAP_THRESHOLD`, they are merged into a single coordinate.
- **Code Reference:** `pipeline.py` -> `close_gaps(vectors)`. This function calculates intersection points of slightly misaligned segments and physically updates the coordinate nodes to overlap.

## 2. Manhattan-World Assumption
Architectural spaces overwhelmingly adhere to orthogonal alignments (walls are 90 degrees or parallel to one another).
- **Learning Point (Basics):** We gently pull walls so they are perfectly vertical or perfectly horizontal relative to the global grid.
- **Learning Point (Advanced):** We evaluate the angle $\theta$ of each vector. If it is within `ANGLE_TOLERANCE` (e.g., 5-10 degrees) of $0$, $\pi/2$, $\pi$, or $3\pi/2$ radians, we force the coordinates to perfectly align on either the X or Y axis.
- **Code Reference:** `pipeline.py` -> `enforce_manhattan(vectors)`. It iterates through all segments, measures `np.arctan2(dy, dx)`, and applies snap thresholds. If a wall is designated "horizontal", it locks $y_1 = y_2$ by averaging their Y coordinates.

## 3. Healing Noise Gaps
- **Learning Point:** Sometimes the U-Net misses a chunk of data inside a wall, breaking one wall into two disjoint collinear walls.
- **Code Reference:** `structural_corrector.py` -> `fill_small_gaps()`. This function evaluates collinear walls (walls on the exact same axis), checks the distance between their adjacent endpoints, and structurally fuses them if the gap is smaller than a noise threshold.

## External Resources & Further Reading
1. **The Manhattan World Assumption:** [Understanding the Manhattan World Assumption in Computer Vision](https://www.semanticscholar.org/paper/The-Manhattan-world-assumption%3A-regularities-in-Coughlan-Yuille/a906ff67de4bc08d2fc1baeb147391eaaf6fe48a)
2. **Euclidean Distances and Geometric Snapping:** [Spatial Indexing with KD-Trees](https://en.wikipedia.org/wiki/K-d_tree)
