# Topic 4: Door Integration and Topological Wall Carving (Core Milestone)

This phase represents the primary breakthrough separating this project from a standard visualization script. To generate a true BIM model, doors cannot simply be "drawn as colored patches over walls." 

**The architectural structure must be natively shattered.** The 2D wall graph vectors must be physically mathematically sliced into disjoint sub-segments wherever a doorway structurally lies.

---

## 1. Probabilistic Wall Selection (MathGPT Multi-Variate Cost)

When YOLO detects a door, it hands us a raw `[x, y]` coordinate. We need to figure out *which wall* the door belongs to. A naive algorithm just searches for the wall mathematically closest to the `[x,y]` center.

### The T-Junction Failure
A door often sits exactly at the intersection of a vertical wall and a horizontal wall. The perpendicular distance from the door to *both* walls is roughly $0.0$. The naive algorithm guesses, resulting in doors aggressively attempting to snap to incorrect perpendicular walls 50% of the time.

### The Multi-Variate Cost Function ($J_{orient}$)
We re-engineered the selection engine using a three-variable weighted cost function:
$$ \text{Cost}_i = \alpha \cdot d_{\text{perp}, i} + \beta \cdot J_{\text{orient}, i} + \gamma \cdot J_{\text{proj}, i} $$

1. **$d_{perp}$ (Distance):** How physically close is the door center to the wall line?
2. **$J_{orient}$ (Orientation):** We extract the aspect ratio from the bounding box. If the box is wider than it is tall, the door is definitively horizontal! 
   $$ J_{\text{orient}} = 1.0 - |\mathbf{\hat{w}}_{dir} \cdot \mathbf{\hat{door}}_{axis}| $$
   If the door is horizontal, and the wall is vertical, the dot product is $0$, resulting in a maximum orientation penalty. We enforce a weight of $\beta = 40.0$.
3. **$J_{proj}$ (Projection Span):** Penalizes doors that physically project beyond the endpoints of the wall they are trying to snap to.

By evaluating `cost = d_perp + 40.0 * j_orient + j_proj`, the algorithm gracefully ignores perpendicular walls automatically without requiring hard-coded exclusionary bounds.

---

## 2. Canonical Center Projections

Once the wall is identified, we must calculate exactly where to cut it.

### The Diagonal Inflation Flaw
Initially, the exact bounding box corners from YOLO were mathematically projected straight down onto the wall. This worked for perfectly Cartesian walls, but if a wall had a $10^\circ$ diagonal tilt, casting the axis-aligned un-rotated YOLO bounding box corners onto the diagonal wall dramatically inflated the projected distance (by a factor of $1/\cos(\theta)$), resulting in massive 2-3 meter gaps for regular doors.

### Strategy 3: Canonical Width Anchors
We overhauled the algorithm to completely ignore YOLO bounding-box corners during slicing operations.
Instead, we extract *only* the single Center Point of the YOLO detection, project it onto the wall vector, and then use predefined physical canonical widths:
- `single_door`: $0.9\text{m}$
- `sliding_door`: $1.2\text{m}$
- `double_door`: $1.6\text{m}$

$$ \text{Cut}_{\text{start}} = p_1 + \max\left(0, \text{proj}_{center} - \frac{\text{Width}_{\text{canonical}}}{2}\right) \cdot \mathbf{\hat{w}}_{dir} $$
This mathematical derivation is inherently immune to angle-inflation and guarantees biologically accurate doorway metrics permanently.

---

## 3. Resolving Silent Corner Skips

When a door physically abuts a corner, its projected gap actually overlaps the very end of the wall vector.

The naive script stated: "If the cut leaves a remaining piece of wall smaller than $2.0px$, the cut is invalid, skip the operation!" This caused doors strictly situated in tight corners to be entirely ignored.

**The Fix:**
We calculate the absolute remaining sizes of the Left Stub ($S_L$) and Right Stub ($S_R$). If a cut on the wall leaves a micro-stub $S_R < 2.0px$, it doesn't mean the cut failed; it explicitly means the doorway physically consumes the entire corner. The algorithm simply *absorbs* the stub and natively drops it during the re-insertion phase.

---

## External Resources & Further Reading
1. **Vector Mathematics & Line Projections:** [Point to Line Projection Vector Mathematics](https://en.wikipedia.org/wiki/Vector_projection)
2. **Computational Geometry Algorithms:** [Distance from a Point to a Line Segment](https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment)
3. **BIM Standards for Openings:** [IFC (Industry Foundation Classes) Openings Logic](https://technical.buildingsmart.org/)
