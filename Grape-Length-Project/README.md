# User’s Manual

## 1. Overview

This code is designed for:
* Training a YOLOv8 object detection model on a custom dataset of grape images. [source: 1]
* Performing inference on time-series images of grapes to measure their diameters over time. [source: 2]
* Producing visualization outputs (e.g., smoothed growth curves, box plots) and saving various metric files. [source: 3]

The code is structured into 11 main parts, each clearly labeled. [source: 4] They run in sequence, from environment setup to final inference and visualization. [source: 5] Most parameters and file paths can be customized within these parts. [source: 6]

## 2. Prerequisites

* **Environment:** Google Colab (or any environment with a GPU-capable Python setup). [source: 7]
    * If using Google Colab, the code includes commands to mount Google Drive and install libraries. [source: 8]
    * If using a local environment, adapt the code for your folder structure and potentially modify Google Drive mounting lines. [source: 9]
* **Python 3:** With libraries including `ultralytics` (YOLOv8), `scipy`, `numpy`, `pandas`, `matplotlib`, `torch` (PyTorch), `cv2` (OpenCV), and standard libraries like `os`, `shutil`, `glob`. The code attempts to install some if missing. [source: 10]
* **Dataset:** Images in JPG format. [source: 10]
    * Default location: inside an `images` directory under `base_path`. [source: 11]
* **Annotations:**
    * A COCO-format JSON annotation file (for object detection), assumed to be at `coco_json_path`. [source: 12]
    * `Marklabel.csv` for time-series images, containing scale points (`scale_0cm`, `scale_1cm`). This is optional if not measuring diameters. [source: 12, 13, 14]

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
* `base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"`: Holds dataset images, annotations, and training outputs. [source: 16]
* `time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"`: Holds time-series images and `Marklabel.csv`. [source: 17]
You can change these paths. [source: 18]##

## 4. Step-by-Step Usage Instructions

Instructions for each major code part: [source: 19]

### A. Colab Environment Setup (Part 1) [source: 20]

* **Mount Google Drive:**
    * Uses `drive.mount('/content/drive')`. [source: 20]
* **Check Drive:**
    * Verifies if `/content/drive/MyDrive` exists. [source: 21]
* **Install Dependencies:**
    * Uses `pip install -q ultralytics` and `pip install -q scipy`. [source: 21]
    * Verify installation logs for errors. [source: 22]

### B. Imports and Global Settings (Parts 2 & 3) [source: 23]

* Imports necessary libraries (NumPy, Pandas, PyTorch, etc.).
* Sets global paths:
    * `base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"`
    * `time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"`
* Sets `ENABLE_TRAINING = True`. Change to `False` to skip training. [source: 23]
* Modify `base_path` and `time_series_path` if needed. [source: 24]

### C. Smoothing Functions (Part 4) [source: 25]

* Functions `smooth_diameter` and `force_smooth_diameter` handle outlier removal and smoothing. [source: 25]
* No changes are usually needed here. [source: 26]
* Key parameters (defaults shown): `window=3`, `z_threshold=2.5`, `cutoff_str="2024-10-10"`, `max_drop=0.05`. [source: 27]

### D. Convert COCO to YOLO (Part 5) [source: 28]

* Function: `convert_coco_to_yolo()`
* Uses the COCO JSON file from `coco_json_path`. [source: 28]
* Computes bounding boxes in YOLO txt format from annotations. [source: 29]
* Writes label files into a `labels/` folder under `base_path`. [source: 30]
* Keep this step if you have COCO annotations and want to train YOLO. [source: 31] Skip or modify if you already have YOLO txt labels. [source: 32]

### E. Create dataset.yaml (Part 6) [source: 33]

* Function: `create_dataset_yaml()`
* Looks for JPG images in `base_path/images/`. [source: 33]
* Splits images randomly: 80% train, 20% validation. [source: 34]
* Copies corresponding labels to `train/labels` and `val/labels`. [source: 34]
* Writes `dataset.yaml` (YOLOv8 training config) pointing to these folders. [source: 35]
* Adjust code or paths for different ratios or split methods. [source: 36]

### F. Scale Data Loading & Computation (Part 7) [source: 37]

* `load_scale_points`: Reads `Marklabel.csv` for `scale_0cm` and `scale_1cm` coordinates in time-series images. [source: 37]
* `compute_scale`: Calculates pixel distance between scale points. [source: 38]
* Skip or remove scaling references if not measuring physical sizes (cm). [source: 38, 39]

### G. Training Function (Part 8) [source: 40]

* Function: `run_training()`
* Sets YOLO training parameters in `training_params`. [source: 40]
* Calls `model.train(...)` using parameters like `epochs`, `batch`, `device='cuda'`, `exist_ok=True`, and others (lr0, momentum, augmentation). [source: 41]
* After training, calls `model.val(...)` for validation metrics (precision, recall, mAP). [source: 41]
* Extracts PR-curve data. [source: 42]
* Saves key metrics to `metrics.txt`. [source: 42]
* To skip training, set `ENABLE_TRAINING = False`. [source: 43] Ensure `dataset.yaml` and label conversion are correct if training. [source: 44]

### H. Get Latest Model & Quick Visualization (Part 9) [source: 45]

* **`get_latest_model()`:** Finds the newest `best.pt` from training runs. Raises error if none exists. [source: 45, 46]
* **`predict_and_visualize()`:**
    * Loads the newest model checkpoint. [source: 46]
    * Runs YOLO inference on time-series images (in `time_series_path`) and prints detection results. [source: 47]
    * This is a quick check; it doesn't save images or bounding boxes. [source: 48]
    * Use this to ensure the model works on sample images. [source: 49]

### I. Double-S & Time-Series Detection/Analysis (Part 10) [source: 50]

* Performs detection on time-series images. [source: 50]
* Extracts bounding box widths, converts to real-world diameters (cm) using scale info. [source: 51]
* Smoothes data, optionally fits a “double sigmoid” curve, draws plots. [source: 52]
* **Major steps in `predict_time_series(...)`:** [source: 53]
    * **Inference:** Runs the model on each image. [source: 53]
    * **Diameter Computation:** Converts pixel width to cm if scale points exist. [source: 54]
    * **Smoothing:** Applies `smooth_diameter` and `force_smooth_diameter`. [source: 55]
    * **Plotting:** [source: 55]
        * Line plot: original, smoothed, forcibly smoothed diameters. [source: 55]
        * Optional error bars (±SD or ±SE). [source: 56]
        * Optional double-sigmoid growth curve fit (`show_trend=True`). [source: 56]
        * Box plots across dates with outliers. [source: 57]
    * **Saving Data:** [source: 57]
        * `time_series_trend.png` (line plot)
        * `time_series_boxplot.png` (box plot)
        * CSV files with raw data and box-plot stats. [source: 58]
* **Key parameters in `predict_time_series`:** `show_lineplot`, `show_boxplot`, `show_original`, `show_smoothed`, `show_f_smoothed`, `show_sd`, `show_se`, `show_trend`, `show_peak_date`, `show_phase_lines`, `offset_factor`, `show_95ci`. Adapt as needed. [source: 58]

### J. Main Execution (Part 11) [source: 59]

* The `if __name__=="__main__":` block executes the workflow: [source: 59]
    1.  Sets random seed (`set_random_seed(42)`). [source: 60]
    2.  Converts COCO annotations (`convert_coco_to_yolo()`). [source: 61]
    3.  Creates `dataset.yaml` (`create_dataset_yaml()`). [source: 61]
    4.  Trains YOLOv8 if `ENABLE_TRAINING` is `True` (`run_training()`). [source: 61]
    5.  Runs a quick visualization (`predict_and_visualize()`). [source: 61]
    6.  Performs time-series analysis (`predict_time_series(...)`). [source: 62]
        * Uses `time_series_path`, can specify `marklabel_csv`, `model_path` (defaults to latest), `output_dir` (defaults to `results_time_series`). [source: 59]
    7.  Prints "✅ All processes completed!". [source: 60]
* Comment/uncomment lines if your workflow differs (e.g., only inference or only training). [source: 63]

## 5. Customizing Hyperparameters

* **Inside `run_training()`:** [source: 64]
    * Easily change `epochs`, base model (`model="yolov8s.pt"`), `batch` size. [source: 64]
    * Tune advanced parameters like `lr0`, `momentum`, `mosaic` based on your dataset. [source: 65]
* **Inside `predict_time_series()`:** [source: 66]
    * Adjust `show_...` flags for plots and error bars. [source: 66]
    * Toggle double-s fitting with `show_trend`. [source: 66]
* **Paths:** [source: 67]
    * Edit `base_path`, `time_series_path`, and other path references if your files are located elsewhere. [source: 67]

## 6. Common Questions & Troubleshooting

* **EOFError or _pin_memory_loop error during logs?** [source: 68]
    * Sometimes happens with PyTorch multi-process data loading in Colab. [source: 69]
    * Usually doesn't affect results and can often be ignored. [source: 69, 70]
* **Training never starts or no GPU found?** [source: 70]
    * In Colab: `Runtime > Change runtime type > Hardware accelerator: GPU`. [source: 71]
    * Verify GPU availability with `torch.cuda.is_available()` (should return `True`). [source: 72]
* **How to skip training and only do inference?** [source: 73]
    * Set `ENABLE_TRAINING = False` near the top. [source: 73]
    * The code will run `predict_and_visualize()` and `predict_time_series()` using an existing `best.pt` checkpoint. [source: 74]
* **How to gather measurement data after inference?** [source: 75]
    * `predict_time_series()` saves CSV files (e.g., `time_series_measurements.csv`) in the output directory. [source: 75]
    * Open this file in Excel or another tool to see diameter results. [source: 76]
* **How to train on a different dataset structure?** [source: 77]
    * Adapt `create_dataset_yaml()` or skip it and create your own structure manually. [source: 78]
    * Ensure your `dataset.yaml` points correctly to `train:` and `val:` paths. [source: 79]
    * Pass your `.yaml` file path to `model.train(...)`. [source: 79]

## 7. Final Tips

* Always double-check that paths match your actual folder locations. [source: 80]
* Monitor GPU usage in Colab if training is slow or memory-limited. [source: 81]
* You can modify or remove steps you don't need (e.g., scale calculations if not measuring real-world diameters). [source: 82]
* The code is modular; functions can be reused or replaced independently for different workflows. [source: 83]
