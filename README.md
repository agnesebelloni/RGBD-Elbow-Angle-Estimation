# RGBD-Elbow-Angle-Estimation
3D elbow angle estimation from RGB-D data using point cloud processing, segmentation, ICP registration, and Open3D.

## Overview

This project estimates the elbow flexion angle from RGB-D images.

The pipeline reconstructs 3D models of the arm and forearm from multiple RGB-D views, registers new observations to the reference models using ICP, and estimates the elbow angle by selecting the model with the best registration score.

## Pipeline

The project consists of two main phases.

### Offline phase

- RGB-D acquisition from multiple viewpoints
- Arm and forearm segmentation
- Point cloud cleaning
- Point cloud fusion
- Construction of reference elbow models at different angles

### Online phase

- RGB-D acquisition
- Point cloud preprocessing
- ICP registration against the reference models
- RMSE evaluation
- Selection of the estimated elbow angle

## Project Structure

```text
OFFLINE/
    RGBD acquisition
    Segmentation
    Point cloud cleaning
    Point cloud fusion
    Elbow model generation

ONLINE/
    Point cloud preprocessing
    ICP registration
    Angle estimation

results/
report.pdf
README.md
```

## Technologies

- Python
- Open3D
- OpenCV
- NumPy
- Azure Kinect RGB-D camera

## Method

The offline phase generates clean reference models of the arm and forearm at different elbow angles.

During the online phase, an observed RGB-D point cloud is segmented and registered to each reference model using the Iterative Closest Point (ICP) algorithm.

The elbow angle is estimated by selecting the model with the lowest registration error (RMSE).

## Results

The system successfully estimates the elbow angle by comparing the observed point cloud with four reference configurations (0°, 30°, 60°, and 90°).

ICP registration achieved high fitness values and low RMSE, allowing reliable angle estimation.
