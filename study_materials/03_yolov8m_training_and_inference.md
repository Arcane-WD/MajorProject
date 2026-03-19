# Topic 3: YOLOv8m Training and Object Detection

Detecting doors, tables, and beds is an object detection problem. We use **YOLOv8** (You Only Look Once), specifically the *Medium* (YOLOv8m) variant.

## 1. Object Detection Architecture
- **Learning Point (Basics):** Object detection doesn't just classify an image; it draws a "Bounding Box" around the exact spatial location of the object and labels it.
- **Learning Point (Advanced):** YOLO separates the image into an $S \times S$ grid. Each grid cell predicts bounding boxes, confidence scores, and class probabilities simultaneously, making it incredibly fast.
- **Code Reference:** `structural_corrector.py` -> `run_yolo_inference()`. The image is passed through the `ultralytics` YOLO model. YOLO outputs tensors defining `[x_center, y_center, width, height]` for every detected object, which are then parsed into absolute pixel coordinates.

## 2. Supervised Training (mAP50)
- **Learning Point (Basics):** We trained the YOLO model on a custom dataset of thousands of floorplan images where human annotators explicitly drew rectangles over doors and furniture.
- **Learning Point (Advanced):** Our Model achieved an **mAP50 of 0.80**. 
  - **Intersection over Union (IoU):** The overlap between the model's bounding box and the true bounding box.
  - **mAP50 (Mean Average Precision):** The average precision across all classes where a prediction is considered "correct" if its IoU with the ground-truth is $\ge 50\%$.
  - YOLO handles scaling implicitly during training, allowing it to generalize to unseen floorplans.

## 3. Pixel Scaling and Pad Adjustments
- **Learning Point (Advanced):** Computer Vision pipelines (like our U-Net running in 512px Fast Mode) often squish or pad images. YOLO predictions map to the *original* image size.
- **Code Reference:** `structural_corrector.py` -> `correct_structure()`. We implement floating-point affine transformations to mathematically shrink the YOLO bounding boxes into the $512\times512$ space using `scale = 512.0 / max(h, w)` and dynamically mapping `pad_left` and `pad_top`.

## External Resources & Further Reading
1. **YOLOv8 Architecture:** [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. **Understanding mAP Evaluation Metrics:** [Mean Average Precision (mAP) in Object Detection](https://blog.roboflow.com/mean-average-precision/)
3. **Bounding Box IoU:** [Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
