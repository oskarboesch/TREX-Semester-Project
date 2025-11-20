from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from scipy.io import loadmat
import cv2
import torch

def list_logs(path_data, dts_45=None):
    """
    Creates a table (DataFrame) for each measurement log, for easy filtering.
    
    Args:
        path_data (str or Path): Path to the root data folder (experiment folder)
        dts_45 (list of datetime, optional): Experiments with guide 45°
    
    Returns:
        pd.DataFrame: logs table with extracted information
    """
    path_data = Path(path_data)
    # Recursively list all CSV files
    log_files = list(path_data.rglob("*.csv"))
    
    logs = []
    
    for file in log_files:
        log_dict = {}
        log_dict["path"] = str(file)
        log_dict["path_markers"] = str(file.parent / "markers.mat")
        # Extract datetime from filename
        match = re.search(r'(?<=log_)([-_0-9]{6,})', str(file))
        if match:
            try:
                log_dict["datetime"] = datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M-%S")
            except:
                log_dict["datetime"] = pd.NaT
        else:
            log_dict["datetime"] = pd.NaT
        
        # Clot index
        match = re.search(r'clot[ _]*(\d+)|(\d+)(?:st|nd|rd|th)[ _\w]*clot', str(file), re.IGNORECASE)

        if match:
            # match.group(1) or match.group(2) may contain the number
            if match.group(1):
                log_dict["clot_index"] = int(match.group(1))
            elif match.group(2):
                log_dict["clot_index"] = int(match.group(2))
            else:
                log_dict["clot_index"] = np.nan
        else:
            log_dict["clot_index"] = np.nan
        
        # Direction
        if re.search("backward", str(file), re.IGNORECASE):
            log_dict["direction"] = "Backward"
        elif re.search("forward", str(file), re.IGNORECASE):
            log_dict["direction"] = "Forward"
        else:
            log_dict["direction"] = np.nan
        
        # Hardness / stiffness
        if re.search("hard", str(file), re.IGNORECASE):
            log_dict["stiffness"] = "Hard"
        elif re.search("medium", str(file), re.IGNORECASE):
            log_dict["stiffness"] = "Medium"
        elif re.search("soft", str(file), re.IGNORECASE):
            log_dict["stiffness"] = "Soft"
        else:
            log_dict["stiffness"] = np.nan
        
        # Pressure
        if re.search("low", str(file), re.IGNORECASE):
            log_dict["pressure"] = "Low"
        elif re.search("high", str(file), re.IGNORECASE):
            log_dict["pressure"] = "High"
        else:
            log_dict["pressure"] = np.nan
        
        # Clot presence
        if re.search("without", str(file), re.IGNORECASE):
            log_dict["clot_presence"] = "Without"
        else:
            log_dict["clot_presence"] = "With"
        
        # Guide 45°
        if dts_45 is not None and pd.notna(log_dict["datetime"]):
            log_dict["guide_straight"] = log_dict["datetime"].date() not in [d.date() for d in dts_45]
        
        logs.append(log_dict)
    
    logs_df = pd.DataFrame(logs)
    
    # Sort by datetime
    logs_df = logs_df.sort_values("datetime").reset_index(drop=True)
    
    return logs_df




def load_data(logs, heads_keep=None, heads_rename=None, fss=None):
    """
    Loads data for all logs as a list of DataFrames.
    
    Args:
        logs (pd.DataFrame): Must contain a 'path' column
        heads_keep (list of str, optional): Columns to keep
        heads_rename (list of str, optional): New column names
        fss (float, optional): Force sensor voltage to mN slope factor
    
    Returns:
        list of pd.DataFrame
    """
    data = []
    
    for idx, row in logs.iterrows():
        # Load CSV data
        df = pd.read_csv(row["path"], sep=';')
        # Unwrap time and sync timestamps
        timestamps = df["timestamp [s]"].diff().fillna(0).cumsum()
        frame0_idx = df[df.get("Camera_trigger_signal", False)].index[0] if "Camera_trigger_signal" in df else 0
        df["timestamp [s]"] = timestamps - timestamps.iloc[frame0_idx]
        
        # Keep only selected columns
        if heads_keep is not None:
            df = df[heads_keep]

        # Rename columns
        if heads_rename is not None:
            df.columns = heads_rename
        
        # Convert force to mN
        if fss is not None and "force_sensor_v" in df.columns:
            df["force_sensor_mN"] = df["force_sensor_v"] * fss

        
        data.append(df)

        # check if data is empty
        if df.empty:
            print(f"Warning: DataFrame for log index {idx} is empty after loading.")
    
    return data

def load_labels(logs):
    """
    Loads marker data for all logs as a list of dicts.
    
    Args:
        logs (pd.DataFrame): Must contain a 'path_markers'
    Returns:
        labels (list of pd.DataFrame): DataFrame with marker Start and End times for each log
    """
    labels = []

    for idx, row in logs.iterrows():
        # Load marker data
        # Only extract value for keys 'Start' and 'End'
        mat = loadmat(row["path_markers"])
        # Start position
        start_pos = mat['Start'][0, 0] 

        # End position
        end_pos = mat['End'][0, 0]
        label_dict = {
            "Start": float(start_pos),
            "End": float(end_pos)
        }
        labels.append(pd.DataFrame(label_dict, index=[idx]))

    return labels


def load_and_preprocess(log, max_frames=200, threshold=0.1):
    """
    log: pd.DataFrame row with 'path' to the log folder
    Returns:
        - cam1_tensor: torch.Tensor[N, C, H, W]
        - cam2_tensor: torch.Tensor[N, C, H, W]
    """
    folder = Path(log["path"]).parent

    # --- Load calibration images RGB ---
    calib1_rgb = cv2.imread(str(folder / "calibration_image.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    calib2_rgb = cv2.imread(str(folder / "calibration_imageII.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0


    # plot calibration images
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Calibration Image Camera 1")
    # show the images with matplotlib
    plt.imshow(calib1_rgb[...,::-1])  # Convert BGR to RGB
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Calibration Image Camera 2")
    plt.imshow(calib2_rgb[...,::-1])  # Convert BGR to RGB
    plt.axis('off')
    plt.show()

    # --- Load calibration images ---
    calib1 = cv2.imread(str(folder / "calibration_image.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    calib2 = cv2.imread(str(folder / "calibration_imageII.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # --- Load frame lists ---
    cam1_paths = sorted(folder.glob("frameID_*.jpg"))[:max_frames]
    cam2_paths = sorted(folder.glob("II_frameID_*.jpg"))[:max_frames]

    # Preallocate arrays
    cam1_imgs = np.stack([
        (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 - calib1) > threshold
        for p in cam1_paths
    ]).astype(np.float32)

    cam2_imgs = np.stack([
        (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 - calib2) > threshold
        for p in cam2_paths
    ]).astype(np.float32)

    # Convert to tensors: [N, C, H, W]
    cam1_tensor = torch.from_numpy(cam1_imgs).unsqueeze(1)
    cam2_tensor = torch.from_numpy(cam2_imgs).unsqueeze(1)


    return cam1_tensor, cam2_tensor


