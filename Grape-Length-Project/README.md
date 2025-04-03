# Grape-Length-Project

A simple guide and reference for training a YOLOv8 model on grape images, measuring grape diameters over time, and generating visual plots. This project includes:

- **Python code** for dataset splitting, training, and inference.
- **Time-series analysis** of grape diameters (including smoothing and double-sigmoid fitting).
- **Plots** (line plots and box plots) to visualize the growth trend.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [File and Folder Layout](#file-and-folder-layout)
4. [Step-by-Step Usage Instructions](#step-by-step-usage-instructions)
5. [Customizing Hyperparameters](#customizing-hyperparameters)
6. [Common Questions & Troubleshooting](#common-questions--troubleshooting)
7. [Final Tips](#final-tips)

---

## Overview
This repository contains:
- **Code** for training YOLOv8 on grape images.
- **Time-series inference** to measure grape diameters by analyzing bounding box widths with scale references.
- **Visualization** of growth trends (smoothed line plots, box plots, double-sigmoid fitting, etc.).

You can train the model yourself or skip training if you only want inference. The code is split into clearly labeled sections for easy navigation.

---

## Prerequisites
1. A **Python 3** environment (e.g., Google Colab, local machine with GPU, or a similar setup).
2. Installed libraries:
   - ultralytics
   - scipy
   - numpy
   - pandas
   - matplotlib
   - torch
   - opencv-python
3. Dataset of JPG images and a COCO-format JSON annotation file.
4. (Optional) Marklabel.csv for physical diameter scaling from images.

---

## File and Folder Layout

```
Grape-Length-Project/
├── Training_Pics/
│   ├── images/
│   ├── labels/
│   ├── labels_my-project-name_2025-03-06-03-02-19.json
├── Time_Series_Sample_K5/
│   ├── Marklabel.csv
│   ├── (time-lapse images)
├── notebooks/
│   └── grape_length_analysis.py
├── results/
├── README.md
```

---

## Step-by-Step Usage Instructions

Detailed instructions on running, training, and visualizing results are provided inside `grape_length_analysis.py`. You can run this in Google Colab or locally.

---

## Customizing Hyperparameters

Edit the `training_params` dictionary inside the script to customize YOLO training settings.

---

## Common Questions & Troubleshooting

See inline comments and printed logs inside the script. Most steps have clear success/failure printouts.

---

## Final Tips

- Ensure your image names contain valid dates for time-series plotting (e.g., `20240620_sample.jpg`).
- The double-sigmoid model requires sufficient number of valid data points across dates.