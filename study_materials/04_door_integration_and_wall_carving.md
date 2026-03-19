# Topic 4: Door Integration and Topological Wall Carving (Core Milestone)

This was one of the most mechanically complex phases of the project. To generate a true BIM model, doors cannot simply be "drawn over" walls. The 2D wall graph vectors must be mathematically shattered (sliced) where the door intersects them.

## 1. Probabilistic Wall Selection
When YOLO detects a door, it gives an XY coordinate. We need to figure out *which wall* the door belongs to.
- **Learning Point (Basics):** We find the shortest perpendicular distance between the door's center and all nearby wall vectors.
- **Learning Point (Advanced):** At T-junctions, the perpendicular distances to two intersecting walls are mathematically identical. To prevent snapping to the wrong wall, we implement an **Orientation Heuristic**. 
- **Code Reference:** `structural_corrector.py` -> `carve_doors_out_of_walls()`. We calculate the aspect ratio of the YOLO bounding box (`det["width"] / det["height"]`). If the door is wider than it is tall, it strongly implies the wall it sits on is horizontal. If the nearest wall evaluates as completely vertical, we impose an artificial $+40.0\text{m}$ cost penalty, mathematically preventing improper snaps.

## 2. Mathematical Projection and Carving
Once the ideal wall vector (from point $p_1$ to $p_2$) is selected, we must slice a physical gap out of it.
- **Learning Point (Basics):** A projection calculates where a point in space (the door center) lands straight down onto a line (the wall).
- **Learning Point (Advanced):** Early project iterations mapped the 4 corners of the YOLO box. This failed dynamically because diagonally-aligned bounding boxes drastically inflated the projected door widths (due to cosine projection distortions).
- **Strategy 3 Implementation:** We shifted to relying entirely on the **central point ($C$)** and canonical physical widths (e.g., exactly $1.6\text{m}$ for double doors). 
  - The gap edges are strictly evaluated symmetrically from the projected center: $ \text{cut}_{\text{start, end}} = C \pm (\text{width} / 2) $.
- **Code Reference:** `structural_corrector.py`. The algorithm extracts `proj_min` and `proj_max` strictly from the center offset. It then validates the remaining Left and Right stubs ($S_L$, $S_R$). If a door spans directly to a corner node ($S_L < 2.0\text{px}$), the algorithm autonomously consumes the entire corner stub physically.

## External Resources & Further Reading
1. **Vector Mathematics & Line Projections:** [Point to Line Projection Vector Mathematics](https://en.wikipedia.org/wiki/Vector_projection)
2. **Computational Geometry Algorithms:** [Distance from a Point to a Line Segment](https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment)
3. **BIM Standards for Openings:** [IFC (Industry Foundation Classes) Openings Logic](https://technical.buildingsmart.org/)
