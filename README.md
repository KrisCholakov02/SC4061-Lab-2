# SC4061 Lab 2: Image Segmentation & 3D Stereo Vision

## Overview

This repository contains the MATLAB code and documentation for a laboratory project in Computer Vision. The project explores fundamental concepts through practical experiments focused on image segmentation and 3D stereo vision.

## Purpose

The primary objective of this project is to investigate and compare various techniques in computer vision, with a focus on two main areas:

- **Image Segmentation:**  
  The experiments evaluate several segmentation techniques to effectively isolate text from degraded document images. The methods include:
  - **Otsu’s Global Thresholding:** A method that computes an optimal single threshold based on maximizing inter-class variance.
  - **Niblack’s Local Thresholding:** A local approach that adapts the threshold to each pixel based on local mean and standard deviation, enhanced with Bayesian optimization for parameter tuning.
  - **Sauvola’s Thresholding:** An improvement on Niblack’s method that adjusts the threshold further by incorporating a dynamic range parameter.
  - **Filter Bank with K-Means Clustering:** A technique that combines Gabor filter-based texture feature extraction with K-Means clustering to segment text regions, particularly useful for noise reduction.

- **3D Stereo Vision:**  
  The project also explores methods for computing disparity maps from stereo image pairs (both synthetic and real), which are essential for depth perception. Techniques applied include:
  - **Sum of Squared Differences (SSD)**
  - **Normalized Cross-Correlation (NCC) and Zero-Mean NCC (ZNCC)**
  - **Sum of Absolute Differences (SAD)**
  
  These methods aim to match corresponding points between left and right images to extract depth information, highlighting the challenges of matching in areas with low texture or repetitive patterns.

## Project Details

### Image Segmentation
- **Otsu’s Method:** Utilizes global image statistics to determine a single threshold that separates foreground (text) from background.
- **Niblack’s Method:** Computes local thresholds by considering the mean and standard deviation within a neighborhood. Bayesian optimization is used to fine-tune parameters such as window size and the sensitivity factor \( k \).
- **Sauvola’s Method:** Further refines local thresholding by introducing a dynamic range parameter \( R \) to better handle noise and complex backgrounds.
- **Filter Bank & K-Means:** Employs a set of Gabor filters to capture directional texture features. The features are then clustered with K-Means to identify text regions.

### 3D Stereo Vision
- **Disparity Map Computation:**  
  Different similarity metrics (SSD, NCC/ZNCC, and SAD) are applied to compute disparity maps that represent depth.  
- **Depth Estimation:**  
  The disparity maps are used to infer depth, where higher disparities correspond to closer objects and lower disparities indicate objects that are farther away. This section discusses the challenges of textureless or repetitive regions and compares the effectiveness of each matching method.

## Tools and Environment

- **MATLAB:** The project is implemented in MATLAB, leveraging the Image Processing Toolbox and the Statistics and Machine Learning Toolbox.
- **Bayesian Optimization:** Used to optimize parameters in local thresholding methods for improved segmentation accuracy.
- **Image Data:** The document images and stereo pairs are stored in the `img` directory.

## Results and Insights

- **Image Segmentation:**  
  The experiments demonstrate that while global thresholding (Otsu’s method) performs adequately under uniform conditions, local thresholding methods (Niblack’s and Sauvola’s) provide better performance on degraded images with uneven illumination. The filter bank approach offers a robust alternative when noise reduction is critical.
  
- **3D Stereo Vision:**  
  Disparity maps generated using SSD, NCC/ZNCC, and SAD reveal the trade-offs between noise sensitivity and matching accuracy. Correlation-based methods (NCC/ZNCC) tend to offer smoother depth maps compared to SSD, particularly in challenging regions.

## How to Use

1. **Code:**  
   Explore the MATLAB scripts corresponding to each experiment. Each script is well-commented to explain its functionality and the underlying algorithm.

2. **Data:**  
   The `img` directory contains all the image data used for segmentation and stereo vision experiments.

3. **Results:**  
   Running the scripts will generate figures and quantitative evaluations that can be cross-referenced with the detailed report.

## References

- Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization. *Pattern Recognition, 33*(2), 225–236. doi:10.1016/S0031-3203(99)00055-2
