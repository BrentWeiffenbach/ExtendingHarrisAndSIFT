# ExtendingHarrisAndSIFT
Extending Harris &amp; SIFT from 2D to 3D for CS545 at WPI
# 3D Feature Detection Project

This repository contains implementations and analyses of 3D keypoint detectors on both voxel grids and point clouds, including Harris 3D and SIFT 3D.

---

## 🧩 Part A — Harris Corner Detector in 3D

### A1. 3D Harris on Voxel Grids

**Goal:**  
Extend the 2D Harris corner detector to a 3D volume (scalar field `I(x, y, z)`).

**Steps:**
1. **Compute Gradients:**  
   Calculate derivatives in all three directions (`Ix`, `Iy`, `Iz`) using 3D Sobel filters or derivative-of-Gaussian.
2. **Build 3D Structure Tensor:**  
   Construct the local structure tensor and smooth it with a Gaussian to capture intensity variations in a neighborhood.
3. **Cornerness Score:**  
   Define a cornerness measure using eigenvalues (`λ1 ≥ λ2 ≥ λ3`).  
   Example formula:  R = det(M) - k * trace(M)^3
   - `k` typically between 0.04 and 0.1  
  - Strong corners → large variation in all 3 directions
4. **3D Non-Maximum Suppression (NMS):**  
Keep only local maxima within a 3D neighborhood.
5. **Visualization:**  
Overlay detected points on slices, isosurfaces, or volume renderings.

**Deliverables:**
- Explanation of design choices (filters, σ, k, NMS window)  
- Implementation + parameter values  
- Visual results with captions  
- Discussion on smoothing scale, sampling resolution, and anisotropy

---

### A2. 3D Harris on Point Clouds

**Goal:**  
Adapt Harris-like detection to unstructured data (point clouds).

**Steps:**
1. Define neighborhoods using either k-nearest neighbors (k-NN) or radius-based neighborhoods.
2. Compute local covariance / structure tensor from neighbors.
3. Define cornerness measure using eigenvalues.
4. Apply NMS to suppress nearby points within a radius.

**Deliverables:**
- Description of neighborhood and tensor computation  
- Cornerness function + thresholds  
- Keypoint visualizations on the point cloud  
- Analysis of sensitivity to point density, noise, and outliers

---

## 🧭 Part B — SIFT Keypoint Detection in 3D

### B1. 3D SIFT on Voxel Grids

**Goal:**  
Extend SIFT scale-space detection to 3D volumes.

**Steps:**
1. Build a 3D Gaussian pyramid `G(x, y, z, σ)`.
2. Compute Difference of Gaussians (DoG) by subtracting adjacent scales.
3. Detect scale-space extrema by comparing each voxel with:
- 26 neighbors in space  
- Neighbors in adjacent scales
4. (Optional) Refine keypoints for better location and scale accuracy.
5. Discuss orientation: how to define 3D gradients/orientations.

**Deliverables:**
- Method description + parameters (octaves, levels per octave, σ schedule)  
- Implementation of extrema detection  
- Visualizations  
- Discussion on memory usage and computational cost

---

### B2. SIFT-like Detection on Point Clouds

**Goal:**  
Create a multi-scale keypoint detector for point clouds.

**Scale-space options:**
- Increase neighborhood radius  
- Convert to voxel grid (resampling)  
- Graph-based diffusion (advanced/bonus)

**Steps:**
1. Build scale-space representation
2. Define a DoG-like operator
3. Detect extrema across scales
4. Map detections back to original points

**Deliverables:**
- Method choice + justification  
- Implementation + parameters  
- Visualizations  
- Comparison with voxel-based SIFT (B1)

---

## 📊 Part C — Evaluation & Analysis

**Datasets:**  
Use multiple datasets, including at least one synthetic dataset.

**Quantitative Metrics:**
- Repeatability under rotation, noise, downsampling  
- Stability of keypoint counts  
- Localization error

**Qualitative Analysis:**
- Visual comparisons on volumes and point clouds

**Comparisons:**
- Grid-based vs point cloud methods  
- Harris 3D vs SIFT 3D  
- Parameter sensitivity: k, σ, neighborhood size

**Additional Requirements:**
- Plots and/or tables summarizing results  
- Clear experimental setup for reproducibility

---
