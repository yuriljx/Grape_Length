User’s Manual

1. Overview

This code is designed for:

Training a YOLOv8 object detection model on a custom dataset of grape images.

Performing inference on time-series images of grapes to measure their diameters over time.

Producing visualization outputs (e.g., smoothed growth curves, box plots) and saving various metric files.

The code is structured into 11 main parts, each clearly labeled. They run in sequence, from environment setup to final inference and visualization. Most parameters and file paths can be customized within these parts.

2. Prerequisites

Google Colab (or any environment with a GPU-capable Python setup).

If you are running this in Google Colab, the code already includes commands to mount Google Drive and install required libraries.

If you are running in a local environment, you will need to adapt the code to your own folder structure and possibly remove or modify the Google Drive mounting lines.

Python 3 environment with the following libraries (the code automatically installs some if missing):

ultralytics (which includes YOLOv8)

scipy

numpy

pandas

matplotlib

torch (PyTorch)

cv2 (OpenCV)

and a few more standard libraries (e.g., os, shutil, glob, etc.)

A dataset of images in JPG format.

By default, images should be located inside a directory named images under base_path.

A COCO-format JSON annotation file (for object detection) is assumed to be placed in coco_json_path.

Marklabel.csv for time-series images**.

This file holds the information about scale points (scale_0cm and scale_1cm) on the images.

If you do not have such a file, or do not need to measure grape diameters in a time-lapse manner, you can skip the relevant steps.

3. File and Folder Layout

By default, the code references the following structure inside a Google Drive folder (you can adapt paths if needed):

YourDrive/

└─ DeepLearning_Projects/

└─ Grape_Length/

├─ Training_Pics/

│   ├─ images/

│   ├─ labels_my-project-name_2025-03-06-03-02-19.json

│   ├─ ...other files...

└─ Time_Series_Sample_K5/

├─ Marklabel.csv

├─ images relevant to time-lapse

└─ ...

Key paths:

base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"

This folder holds your primary dataset images (in images/), your annotation JSON file, and any outputs from YOLO training.

time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"

This folder holds time-series images of grapes, plus Marklabel.csv for scale points.

You can change these paths based on your own environment.

4. Step-by-Step Usage Instructions

Below are instructions for each major part of the code.

A. Colab Environment Setup (Part 1)

Mount Google Drive

drive.mount('/content/drive')

This connects your Google Drive to the Colab session, allowing direct file access.

Check Drive

if os.path.exists("/content/drive/MyDrive"):

print("✅ Google Drive successfully mounted")

else:

print("❌ Google Drive not mounted, please re-run drive.mount() if needed")

Install Dependencies

!pip install -q ultralytics

!pip install -q scipy

print("✅ Ultralytics & scipy installation finished")

This ensures YOLOv8 (ultralytics) and scipy are installed.

No additional actions needed here except verifying the installation logs for any errors.

B. Imports and Global Settings (Parts 2 & 3)

The code imports all necessary Python libraries (NumPy, Pandas, PyTorch, etc.) and sets up a few global paths:

base_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Training_Pics"

time_series_path = "/content/drive/MyDrive/DeepLearning_Projects/Grape_Length/Time_Series_Sample_K5"

...

ENABLE_TRAINING = True

If you prefer not to train (and only want inference), you can set

ENABLE_TRAINING = False.

You can change base_path and time_series_path to match your own file locations.

C. Smoothing Functions (Part 4)

Two functions, smooth_diameter and force_smooth_diameter, handle outlier removal and smoothing of grape diameters over time. Unless you need to modify how smoothing is done, you do not need to change these.

Key parameters:

window=3 for rolling average

z_threshold=2.5 for outlier detection

cutoff_str="2024-10-10" and max_drop=0.05 for restricting diameter drops post-certain date

Most users do not need to modify these defaults.

D. Convert COCO to YOLO (Part 5)

Function: convert_coco_to_yolo()

It takes the COCO-format JSON file specified by coco_json_path.

Loops through each annotation to compute bounding boxes in YOLO txt format.

Writes the new label files into a labels/ folder under base_path.

If you have a COCO-format annotation file and want to train YOLO, keep this step. Otherwise, if you already have YOLO txt labels, you can skip or modify it.

E. Create dataset.yaml (Part 6)

Function: create_dataset_yaml()

Looks inside base_path/images/ for JPG images.

Randomly splits them into a train set (80%) and val set (20%).

Copies corresponding labels into train/labels and val/labels.

Writes dataset.yaml (training config for YOLOv8) pointing to these folders.

This is helpful for an automatic train/val split. If you prefer a different ratio or have a specific split method, you can adjust the code or the paths.

F. Scale Data Loading & Computation (Part 7)

load_scale_points reads a CSV (Marklabel.csv) that contains coordinates for scale_0cm and scale_1cm in each time-series image.

compute_scale calculates the pixel distance between these two points. If your time-lapse images do not have scale points, you can skip or remove references to scaling.

This is important only if you want to measure physical sizes (in cm) in your images.

G. Training Function (Part 8)

Function: run_training()

Sets up various YOLO training parameters in training_params.

Calls model.train(...) from the Ultralytics YOLO library, using:

epochs: total training epochs

batch: batch size

device: 'cuda' if using GPU

exist_ok=True: to allow overwriting existing folders

Other advanced hyperparameters like lr0, momentum, data augmentation settings, etc.

After training, calls model.val(...) to run YOLO’s built-in validation and compute metrics (precision, recall, mAP, etc.).

Extracts PR-curve data from metrics.

Saves key metrics to a file at metrics.txt.

If you do not want to train, simply set ENABLE_TRAINING = False. Otherwise, ensure your dataset.yaml and COCO → YOLO conversion are set up properly.

H. Get Latest Model & Quick Visualization (Part 9)

Two utility functions:

get_latest_model()
Finds the newest best.pt file from the training runs. If none exists, raises an error.

predict_and_visualize()

Loads the newest model checkpoint.

Scans through time-series images (in time_series_path) and runs YOLO inference on each one, printing detection results.

Does not save any inference images or bounding boxes — purely for a quick check.

Use predict_and_visualize() to ensure your model is working as expected on sample images.

I. Double-S & Time-Series Detection/Analysis (Part 10)

This is the code that:

Performs detection on each time-series image.

Extracts bounding box widths, uses the scale information to convert them into real-world diameters (in cm).

Smoothes the data, optionally fits a “double sigmoid” curve, and draws line plots and box plots.

Major steps within predict_time_series(...):

Inference: runs the trained model on each image to get bounding boxes.

Diameter Computation: if scale points are found, the bounding box width (in pixels) is converted to centimeters.

Smoothing: applies smooth_diameter and force_smooth_diameter.

Plotting:

A line plot showing original, smoothed, and forcibly smoothed diameters.

Optional error bars (±SD or ±SE).

Double-sigmoid (if show_trend=True) for a growth curve fit.

Box plots across dates, with outliers identified.

Saving Data:

time_series_trend.png: the line plot.

time_series_boxplot.png: the box plot.

CSV files containing raw measurement data and box-plot statistics.

Key parameters in predict_time_series:

show_lineplot, show_boxplot (whether to display each plot)

show_original, show_smoothed, show_f_smoothed (which lines to plot)

show_sd, show_se (type of error bars)

show_trend (whether to fit double-s curve)

show_peak_date (draw a vertical line at the largest diameter)

show_phase_lines (draw lines for double-s transitions)

offset_factor (affects the distance of phase lines)

show_95ci (show shading for ±95% confidence intervals, if error bars are enabled)

You can adapt these arguments to your needs.

J. Main Execution (Part 11)

Finally:

if __name__=="__main__":

set_random_seed(42)

convert_coco_to_yolo()

create_dataset_yaml()

if ENABLE_TRAINING:

run_training()

predict_and_visualize()

predict_time_series(

sample_folder=time_series_path,

marklabel_csv=None,  # or specify your CSV if needed

model_path=None,     # loads the latest best.pt by default

output_dir=None,     # will create a results_time_series folder automatically

...

)

print("✅ All processes completed!")

The main function does everything in sequence:

Sets the random seed.

Converts the COCO annotation and creates dataset.yaml.

If ENABLE_TRAINING = True, trains YOLOv8.

Runs predict_and_visualize() for a quick check.

Finally calls predict_time_series(...) to do the advanced diameter measurement and plotting.

If your workflow is only about inference or only about training, you can comment/uncomment lines as needed.

5. Customizing Hyperparameters

Inside run_training():

You can change the number of epochs (epochs), the base YOLO model (model="yolov8s.pt"), or the batch size (batch) easily. Further advanced parameters (like lr0, momentum, mosaic, etc.) can be tuned based on your dataset.

Inside predict_time_series():

Adjust the show_... flags to control which lines and error bars are plotted. Toggle double-s fitting with show_trend.

Paths:

If you want to store your outputs in a different location or have your images somewhere else, edit base_path, time_series_path, and references to them throughout the code.

6. Common Questions & Troubleshooting

Why do I see an EOFError or _pin_memory_loop error in the logs?

This sometimes occurs when PyTorch uses multi-process data loading in Colab. It typically does not affect training/inference results. You can often safely ignore it.

What if my training never starts or no GPU is found?

In Google Colab, go to Runtime > Change runtime type > Hardware accelerator: GPU. Ensure your session actually sees the GPU by checking torch.cuda.is_available() returns True.

How do I skip training and only do inference?

Set ENABLE_TRAINING = False near the top. The rest of the code will still run predict_and_visualize() and predict_time_series() using an existing best.pt checkpoint.

How do I gather measurement data after inference?

The predict_time_series() function saves CSV files (e.g. time_series_measurements.csv) in the output directory. This contains the diameter results and can be opened in Excel or any other tool.

I want to train on a different dataset structure. What do I change?

Adapt create_dataset_yaml() or skip it and manually create your own dataset structure. Make sure your dataset.yaml points to the correct paths for train: and val:. Then pass that .yaml to model.train(...).

7. Final Tips

Always check your paths to ensure they match your actual folder locations.

Monitor GPU usage in Colab if training is slow or memory-limited.

You can rename or remove steps you don’t need, such as the scale calculations if you aren’t measuring real-world diameters.

This code is modular: each function can be reused or replaced independently if you have a different workflow in mind.

