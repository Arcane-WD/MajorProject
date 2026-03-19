# Topic 1: U-Net Mask Generation & Vector Extraction

The journey from a raster image (pixels) to a mathematical Building Information Model (BIM) begins with isolated semantic segmentation followed by vectorization. This phase bridges the gap between raw unstructured data (an image) and structured geometric relationships (graphs).

---

## 1. Deep Learning Segmentation (U-Net)

Before we can measure walls, we need the computer to understand which pixels represent structural boundaries vs empty floor space.

### The Architecture: U-Net with ResNet-34
- **What is it?** U-Net is a convolutional neural network (CNN) originally developed for biomedical image segmentation. It features an encoder (contracting path) and a decoder (expanding path), giving it a U-shaped architecture.
- **Why ResNet-34?** We use ResNet-34 as the encoder backbone because its residual connections (skip connections between layers) solve the vanishing gradient problem, allowing the network to learn rich, deep features without losing spatial accuracy.
- **Skip Connections:** The defining feature of U-Net. As the decoder upsamples the image back to its original resolution, it concatenates high-resolution feature maps directly from the encoder. This allows the network to predict *what* an object is (using deep features) and exactly *where* it is (using shallow features).

### Implementation Details (`pipeline.py`)
```python
def load_model_logic(model_path, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    )
    # ... loads best_cleaner_model_v3.pth
```
- **Tiled Inference:** Floorplans are massive. Passing a $4000 \times 4000$ image into a CNN destroys memory. Instead, `predict_tiled()` chops the image into $512 \times 512$ overlapping tiles. 
- **Hann Window Blending:** To prevent harsh seams where tiles overlap, predictions are weighted using a 2D Hann Window (a bell-curve-like filter). Pixels in the center of a tile are trusted more than pixels at the edges, resulting in a seamlessly stitched global mask.

---

## 2. Raster-to-Vector Extraction

Once we have a clean black-and-white mask of the walls, we must convert these thick pixel blobs into 1-dimensional mathematical vectors.

### Step 2a: Skeletonization
- **Concept:** A morphological operation that iteratively strips away the outer boundary of a shape until only the 1-pixel-wide "skeleton" remains.
- **Mathematical Basis:** It relies on the Zhang-Suen thinning algorithm, which evaluates a $3 \times 3$ pixel neighborhood to decide if removing a center pixel will break the connectivity of the line. If it won't break connectivity, it is removed.

### Step 2b: Graph Extraction
A 1-pixel-wide line is still just a bunch of pixels. We need to convert it into a **Graph** $G = (V, E)$ where $V$ are nodes (endpoints/junctions) and $E$ are edges (walls).
- **Node Detection:** 
  - If a pixel has exactly **1 neighbor**, it is a dead-end (Endpoint).
  - If a pixel has **>2 neighbors**, it is an intersection (Junction).
  - If a pixel has exactly **2 neighbors**, it is just a path.
- The algorithm walks pixel-by-pixel between endpoints/junctions to map out the topological edges.

### Step 2c: Ramer-Douglas-Peucker (RDP) Simplification
The graph edges extracted above are jagged (because pixels are squarish). Architectural walls are perfectly straight.
- **The Algorithm:** RDP takes a complex polyline and finds a similar curve with fewer points.
  1. Draw a straight line between the start and end of the polyline.
  2. Find the point on the polyline furthest from this straight line.
  3. If this distance is greater than $\epsilon$ (epsilon), split the polyline at this point and recursively repeat the process for both halves.
  4. If the distance is less than $\epsilon$, discard all intermediate points.
- **Result:** A jagged staircase of 50 pixels is instantly mathematically collapsed into a single line segment defined by just 2 coordinate tuples $(x_1, y_1)$ and $(x_2, y_2)$.

---

## External Resources & Further Reading
1. **U-Net Architecture:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
2. **Skeletonization Algorithms:** [Zhang-Suen thinning algorithm](https://en.wikipedia.org/wiki/Thinning_(morphology))
3. **Ramer-Douglas-Peucker Algorithm:** [RDP Polyline Simplification](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
