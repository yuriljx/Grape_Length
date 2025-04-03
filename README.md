# User’s Manual

## 1. Overview

This code is designed for:
* Training a YOLOv8 object detection model on a custom dataset of grape images. 
* Performing inference on time-series images of grapes to measure their diameters over time. 
* Producing visualization outputs (e.g., smoothed growth curves, box plots) and saving various metric files. 

The code is structured into 11 main parts, each clearly labeled.  They run in sequence, from environment setup to final inference and visualization. Most parameters and file paths can be customized within these parts. 

## 2. Prerequisites

* **Environment:** Google Colab (or any environment with a GPU-capable Python setup). 
    * If using Google Colab, the code includes commands to mount Google Drive and install libraries. 
    * If using a local environment, adapt the code for your folder structure and potentially modify Google Drive mounting lines. 
* **Python 3:** With libraries including `ultralytics` (YOLOv8), `scipy`, `numpy`, `pandas`, `matplotlib`, `torch` (PyTorch), `cv2` (OpenCV), and standard libraries like `os`, `shutil`, `glob`. The code attempts to install some if missing.
* **Dataset:** Images in JPG format.
    * Default location: inside an `images` directory under `base_path`. 
* **Annotations:**
    * A COCO-format JSON annotation file (for object detection), assumed to be at `coco_json_path`. 
    * `Marklabel.csv` for time-series images, containing scale points (`scale_0cm`, `scale_1cm`). This is optional if not measuring diameters. 

## 3. File and Folder Layout

*Default Google Drive structure (adaptable):*

```plaintext
YourDrive/
└─ DeepLearning_Projects/
    └─ Grape_Length/
        ├─ Training_Pics/
        │   ├─ images/           # Training images
        │   └─ labels_*.json     # COCO annotations
        └─ Time_Series_Sample_K5/
            ├─ Marklabel.csv     # Scale points CSV
            └─ images/           # Time-series images
```
**Key paths:**
* `base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"`: Holds dataset images, annotations, and training outputs.
* `time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"`: Holds time-series images and `Marklabel.csv`. 
You can change these paths.##

## 4. Step-by-Step Usage Instructions

Instructions for each major code part: 

### A. Colab Environment Setup (Part 1) 

* **Mount Google Drive:**
    * Uses `drive.mount('/content/drive')`. 
* **Check Drive:**
    * Verifies if `/content/drive/MyDrive` exists. 
* **Install Dependencies:**
    * Uses `pip install -q ultralytics` and `pip install -q scipy`.
    * Verify installation logs for errors.

### B. Imports and Global Settings (Parts 2 & 3)

* Imports necessary libraries (NumPy, Pandas, PyTorch, etc.).
* Sets global paths:
    * `base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"`
    * `time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"`
* Sets `ENABLE_TRAINING = True`. Change to `False` to skip training.
* Modify `base_path` and `time_series_path` if needed.

### C. Smoothing Functions (Part 4)

* Functions `smooth_diameter` and `force_smooth_diameter` handle outlier removal and smoothing.
* No changes are usually needed here.
* Key parameters (defaults shown): `window=3`, `z_threshold=2.5`, `cutoff_str="2024-10-10"`, `max_drop=0.05`. 

### D. Convert COCO to YOLO (Part 5) 

* Function: `convert_coco_to_yolo()`
* Uses the COCO JSON file from `coco_json_path`. 
* Computes bounding boxes in YOLO txt format from annotations. 
* Writes label files into a `labels/` folder under `base_path`. 
* Keep this step if you have COCO annotations and want to train YOLO. Skip or modify if you already have YOLO txt labels.

### E. Create dataset.yaml (Part 6)

* Function: `create_dataset_yaml()`
* Looks for JPG images in `base_path/images/`.
* Splits images randomly: 80% train, 20% validation.
* Copies corresponding labels to `train/labels` and `val/labels`. 
* Writes `dataset.yaml` (YOLOv8 training config) pointing to these folders. 
* Adjust code or paths for different ratios or split methods. 

### F. Scale Data Loading & Computation (Part 7) 

* `load_scale_points`: Reads `Marklabel.csv` for `scale_0cm` and `scale_1cm` coordinates in time-series images. 
* `compute_scale`: Calculates pixel distance between scale points. 
* Skip or remove scaling references if not measuring physical sizes (cm). 

### G. Training Function (Part 8) 

* Function: `run_training()`
* Sets YOLO training parameters in `training_params`. 
* Calls `model.train(...)` using parameters like `epochs`, `batch`, `device='cuda'`, `exist_ok=True`, and others (lr0, momentum, augmentation). 
* After training, calls `model.val(...)` for validation metrics (precision, recall, mAP). 
* Extracts PR-curve data.
* Saves key metrics to `metrics.txt`. 
* To skip training, set `ENABLE_TRAINING = False`. Ensure `dataset.yaml` and label conversion are correct if training. 

### H. Get Latest Model & Quick Visualization (Part 9) 

* **`get_latest_model()`:** Finds the newest `best.pt` from training runs. Raises error if none exists. 
* **`predict_and_visualize()`:**
    * Loads the newest model checkpoint.
    * Runs YOLO inference on time-series images (in `time_series_path`) and prints detection results. 
    * This is a quick check; it doesn't save images or bounding boxes. 
    * Use this to ensure the model works on sample images. 

### I. Double-S & Time-Series Detection/Analysis (Part 10) 

* Performs detection on time-series images. 
* Extracts bounding box widths, converts to real-world diameters (cm) using scale info. 
* Smoothes data, optionally fits a “double sigmoid” curve, draws plots. 
* **Major steps in `predict_time_series(...)`:** 
    * **Inference:** Runs the model on each image. 
    * **Diameter Computation:** Converts pixel width to cm if scale points exist. 
    * **Smoothing:** Applies `smooth_diameter` and `force_smooth_diameter`. 
    * **Plotting:** 
        * Line plot: original, smoothed, forcibly smoothed diameters. 
        * Optional error bars (±SD or ±SE). 
        * Optional double-sigmoid growth curve fit (`show_trend=True`). 
        * Box plots across dates with outliers. 
    * **Saving Data:** [source: 57]
        * `time_series_trend.png` (line plot)
        * `time_series_boxplot.png` (box plot)
        * CSV files with raw data and box-plot stats. 
* **Key parameters in `predict_time_series`:** `show_lineplot`, `show_boxplot`, `show_original`, `show_smoothed`, `show_f_smoothed`, `show_sd`, `show_se`, `show_trend`, `show_peak_date`, `show_phase_lines`, `offset_factor`, `show_95ci`. Adapt as needed. 

### J. Main Execution (Part 11) 

* The `if __name__=="__main__":` block executes the workflow: 
    1.  Sets random seed (`set_random_seed(42)`). 
    2.  Converts COCO annotations (`convert_coco_to_yolo()`). 
    3.  Creates `dataset.yaml` (`create_dataset_yaml()`). 
    4.  Trains YOLOv8 if `ENABLE_TRAINING` is `True` (`run_training()`). 
    5.  Runs a quick visualization (`predict_and_visualize()`). 
    6.  Performs time-series analysis (`predict_time_series(...)`). 
        * Uses `time_series_path`, can specify `marklabel_csv`, `model_path` (defaults to latest), `output_dir` (defaults to `results_time_series`). 
    7.  Prints "✅ All processes completed!". 
* Comment/uncomment lines if your workflow differs (e.g., only inference or only training). 

## 5. Customizing Hyperparameters

* **Inside `run_training()`:**
    * Easily change `epochs`, base model (`model="yolov8s.pt"`), `batch` size. 
    * Tune advanced parameters like `lr0`, `momentum`, `mosaic` based on your dataset. 
* **Inside `predict_time_series()`:** 
    * Adjust `show_...` flags for plots and error bars. 
    * Toggle double-s fitting with `show_trend`. 
* **Paths:** 
    * Edit `base_path`, `time_series_path`, and other path references if your files are located elsewhere. 

## 6. Common Questions & Troubleshooting

* **EOFError or _pin_memory_loop error during logs?** 
    * Sometimes happens with PyTorch multi-process data loading in Colab. 
    * Usually doesn't affect results and can often be ignored. 
* **Training never starts or no GPU found?** 
    * In Colab: `Runtime > Change runtime type > Hardware accelerator: GPU`. 
    * Verify GPU availability with `torch.cuda.is_available()` (should return `True`). 
* **How to skip training and only do inference?** 
    * Set `ENABLE_TRAINING = False` near the top. 
    * The code will run `predict_and_visualize()` and `predict_time_series()` using an existing `best.pt` checkpoint. 
* **How to gather measurement data after inference?** 
    * `predict_time_series()` saves CSV files (e.g., `time_series_measurements.csv`) in the output directory. 
    * Open this file in Excel or another tool to see diameter results. 
* **How to train on a different dataset structure?** 
    * Adapt `create_dataset_yaml()` or skip it and create your own structure manually. 
    * Ensure your `dataset.yaml` points correctly to `train:` and `val:` paths. 
    * Pass your `.yaml` file path to `model.train(...)`. 

## 7. Final Tips

* Always double-check that paths match your actual folder locations. 
* Monitor GPU usage in Colab if training is slow or memory-limited. 
* You can modify or remove steps you don't need (e.g., scale calculations if not measuring real-world diameters). 
* The code is modular; functions can be reused or replaced independently for different workflows. 
