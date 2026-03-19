# Topic 3: YOLOv8m Training and Object Detection

While U-Net is excellent at finding continuous topological regions (walls), it fails to understand *discrete semantic objects*. A BIM model isn't just walls—it needs doors, beds, chairs, and tables. For this, we utilize **YOLOv8** (You Only Look Once), specifically the *Medium* (YOLOv8m) variant.

---

## 1. Object Detection vs Semantic Segmentation

- **Semantic Segmentation (U-Net):** Classifies the image at the pixel level. "This specific pixel is a wall." It has no concept of instances (it doesn't know if a pixel belongs to "Door 1" or "Door 2").
- **Object Detection (YOLO):** Classifies discrete regions. It outputs mathematically defined structures: a Bounding Box `[x_center, y_center, width, height]`, a class label `"double_door"`, and a confidence score `0.85`.

### How YOLO Works
YOLO divides the input image into an $S \times S$ grid. Each grid cell predicts bounding boxes and confidence scores for those boxes. 
- **Anchor Boxes:** YOLO evaluates predefined box shapes (tall boxes, wide boxes) at every grid cell constraint. 
- Because it looks at the entire image exactly one time (a single forward pass through the network), it is exponentially faster than older methods like R-CNN, which relied on analyzing thousands of region proposals individually.

---

## 2. Supervised Training & Evaluation Metrics

We trained the YOLO model on a custom dataset of thousands of architectural floorplan images where human annotators explicitly drew rectangles over specific features.

### Understanding mAP50 (Mean Average Precision)
Our custom network achieved an **mAP50 of 0.80**. What does this mathematically mean?

1. **IoU (Intersection over Union):**
   $$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$
   It is the ratio calculating how perfectly the model's predicted bounding box overlaps with the human's ground-truth bounding box.
2. **Precision & Recall:**
   - **Precision:** Out of all the things the model predicted were "Doors", how many were actually doors?
   - **Recall:** Out of all the real "Doors" in the image, how many did the model successfully find?
3. **mAP50:** When we set the threshold to say "A prediction is correct ONLY if its IoU is $\ge 0.50$", we draw a Precision-Recall curve. The area under this curve is the Average Precision. We then take the Mean of this metric across every single class (doors, beds, toilets) to get `0.80`.

---

## 3. Coordinate Space Normalization (The Scaling Problem)

Computer Vision pipelines are notorious for silent coordinate space mismatches.
- Our U-Net model operates on a padded $512 \times 512$ fixed-resolution array in Streamlit's "Fast Mode".
- YOLOv8 handles scaling natively internally, so when it outputs a bounding box coordinate, it is mapped to the *original raw dimensions* of the user's uploaded image (e.g., $700 \times 550$).

### Floating Point Precision Scaling
If we try to plot a $700\text{px}$ YOLO door directly onto a $512\text{px}$ U-Net wall-map, the door will literally plot off the edge of the screen. We must normalize the YOLO coordinates back down perfectly.

**MathGPT Correction:** Previous iterations used `int()` truncation during scaling. This caused numerical instability. A $511.99$ pixel pad truncated to $511$, slightly shifting the door locations over the walls symmetrically.
We resolved this by retaining native Python `float` operations until the absolute final matrix instantiation.

```python
# In structural_corrector.py -> correct_structure()
scale = 512.0 / max(h, w)
new_w_f = w * scale
new_h_f = h * scale
pad_left_f = (512.0 - new_w_f) / 2.0
pad_top_f  = (512.0 - new_h_f) / 2.0

for det in detections:
    # Maps original image coordinates definitively into the U-Net Padded Space
    det["x_center"] = det["x_center"] * scale + pad_left_f
```

---

## External Resources & Further Reading
1. **YOLOv8 Architecture:** [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. **Understanding mAP Evaluation Metrics:** [Mean Average Precision (mAP) in Object Detection](https://blog.roboflow.com/mean-average-precision/)
3. **Bounding Box IoU:** [Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
