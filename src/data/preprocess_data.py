import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import sosfreqz, butter, zpk2sos, sosfiltfilt
import os
import cv2
from pathlib import Path
from .load_data import get_images_paths_from_log

# Helper functions
def design_butter(f_s, f_c, n, ftype='low'):
    """
    Design a Butterworth filter and return second-order sections.

    Parameters
    ----------
    f_s : float
        Sampling frequency
    f_c : float or list
        Cutoff frequency/frequencies
    n : int
        Filter order
    ftype : str, optional
        Filter type ('low', 'high', 'bandpass', 'bandstop'), default 'low'

    Returns
    -------
    sos : ndarray
        Second-order sections representation
    g : float
        Gain factor
    """
    # Design filter in zpk format
    z, p, k = butter(n, np.array(f_c)/(f_s/2), btype=ftype, output='zpk')
    sos = zpk2sos(z, p, k)
    g = k

    # Stability check
    if np.any(np.abs(p) >= 1):
        print("Warning: Filter may be unstable")


    return sos, g

def crop_data(data: pd.DataFrame, timestamps: np.ndarray, margins: list) -> pd.DataFrame:
    """
    Crop a DataFrame based on start and end margins in seconds.

    Parameters
    ----------
    data : pd.DataFrame
        Table to truncate
    timestamps : np.ndarray
        Array of timestamps corresponding to rows of data
    margins : list
        [start_margin, end_margin] in seconds

    Returns
    -------
    pd.DataFrame
        Cropped DataFrame
    """
    data = data.copy()
    timestamps = np.array(timestamps.copy())
    keep = (timestamps >= timestamps[0] + margins[0]) & (timestamps <= timestamps[-1] - margins[1])
    return data.loc[keep].reset_index(drop=True)

def downsample(df, f_s, f_target):
    """
    Downsample a DataFrame based on desired target sampling frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with timestamps column
    f_s : float
        Original sampling frequency (Hz)
    f_target : float
        Target sampling frequency (Hz)

    Returns
    -------
    pd.DataFrame
        Downsampled data
    """
    df = df.copy()
    factor = int(round(f_s / f_target))
    if factor <= 1:
        return df.reset_index(drop=True)  # No downsampling

    df_down = df.iloc[::factor, :].reset_index(drop=True)

    # recompute timestamps for uniform spacing
    dt = 1.0 / f_target
    df_down["timestamps"] = np.arange(len(df_down)) * dt + df["timestamps"].iloc[0]

    return df_down

def preprocess_logs_old(logs_fit, data_fit, data_fit_plot):
    """
    Preprocess logs: filtering, cropping, downsampling, baseline correction.
    
    Args:
        logs_fit (pd.DataFrame): DataFrame with log metadata
        data_fit (list of pd.DataFrame): Raw data for each log
        data_fit_plot (list of pd.DataFrame): Data for plotting (not downsampled)
    
    Returns:
        None (modifies data_fit and data_fit_plot in place)
    """

    # Parameters
    f_s = 1000  # Sampling rate
    head = "force_sensor_mN"

    # Start preprocessing
    start_time = time.time()

    for i, df in tqdm(enumerate(data_fit)):
        df = reset_timestamps(df, fs=f_s)
        ydata = df[head].values
        df[head] = filter(ydata, sampling_rate=f_s, filter_order=3, cutoff_freq=10)
        # Crop bounds
        direction = logs_fit.loc[i, "direction"].lower()
        if direction == "forward":
            range_crop = [5, 0]
        else:
            range_crop = [2, 0]
        df_cropped = crop_data(df, df["timestamps"], range_crop)
        data_fit_plot[i] = df_cropped.copy()  # Not downsampled

        # Downsample
        df_down = downsample(df_cropped, f_s=f_s, f_target=20)

        # check if df_down is empty after cropping
        if df_down.empty:
            print(f"Warning: DataFrame {i} is empty after downsampling. Skipping preprocessing for this log.")
            data_fit[i] = df_cropped
            continue

        # Baseline correction
        baseline = df_down[head].iloc[0]
        df_down[head] = df_down[head] - baseline
        df_cropped[head] = df_cropped[head] - baseline

        data_fit[i] = df_down
        data_fit_plot[i] = df_cropped

    print(f"Preprocessing time: {time.time() - start_time:.3f} s")

def preprocess_log(df, head, direction, sampling_rate=1000, filter_order=3, downsampling_freq=10, with_bandpower=False):
    df = reset_timestamps(df, fs=sampling_rate)
    lowpass_cutoff = downsampling_freq / 2.0  # Nyquist frequency after downsampling
    data = df[head].values
    df[head] = filter(data, sampling_rate=sampling_rate, filter_order=filter_order, cutoff_freq=lowpass_cutoff)


    # Crop bounds
    if direction == "Forward":
        n = np.random.uniform(1, 8)
        range_crop = [n, 0]
    else:
        range_crop = [2, 0]
    df_cropped = crop_data(df, df["timestamps"], range_crop)
    # Downsample
    df_down = downsample(df_cropped, sampling_rate, downsampling_freq)
    
    # Baseline correction
    baseline = df_down[head].iloc[0]
    df_down[head] = df_down[head] - baseline
    df_cropped[head] = df_cropped[head] - baseline

    if with_bandpower:
        band_powers = compute_band_power(df_down, head, fs=downsampling_freq)
        # need to get rid of window size effect 
        window_size = downsampling_freq // 2
        df_down = df_down.iloc[window_size-1:].reset_index(drop=True)
        assert band_powers.shape[0] == len(df_down), "Band power length mismatch after cropping"
        for i in range(band_powers.shape[1]):
            df_down[f'band_power_{i}'] = band_powers[:, i]

    return df_down

def filter(data, sampling_rate=1000, filter_order=3, cutoff_freq=10):
    sos, _ = design_butter(sampling_rate, cutoff_freq, filter_order, 'low')
    ydata = data - data[0]  # Remove initial value to avoid step
    ydata = sosfiltfilt(sos, ydata)
    return ydata


def compute_band_power(df_down, head, fs, bands=[(0,4),(4,6),(6,10)]):
    """
    df_down: downsampled signal (timestamps must match downsampled points)
    head: signal column
    bands: list of frequency bands
    fs_original: sampling rate of the original signal
    """
    from scipy.signal import spectrogram
    signal = df_down[head].values
    window_size = fs//2  # 0.5 second windows
    overlap = window_size - 1  # 1 sample step
    step_size = window_size - overlap
    num_windows = (len(df_down) - overlap) // step_size
    band_powers_sliding = np.zeros((num_windows, 3))  # 3 bands
    for w in range(num_windows):
        start_idx = w * step_size
        end_idx = start_idx + window_size
        if end_idx > len(signal):
            break
        segment = signal[start_idx:end_idx]
        f, t, Sxx = spectrogram(segment, fs=fs, nperseg=window_size, noverlap=overlap)
        power_spectrum = np.mean(Sxx, axis=1)  # Average over time
        for b, (f_low, f_high) in enumerate(bands):
            band_mask = (f >= f_low) & (f < f_high)
            band_power = np.trapz(power_spectrum[band_mask], f[band_mask])
            band_powers_sliding[w, b] = band_power
    return band_powers_sliding



def preprocess_logs(df_list, head, direction, sampling_rate=1000, filter_order=3, downsampling_freq=10, with_bandpower=False):
    processed_list = []
    for df in df_list:
        processed = preprocess_log(df, head, direction, sampling_rate, filter_order, downsampling_freq, with_bandpower=with_bandpower)
        processed_list.append(processed)
    return processed_list


def get_label_timeseries(labels, logs):
    """
    Compute a binary array for each time steps in logs indicating if in the clot or not based on labels.
    Args:
        labels (list of pd.DataFrame): DataFrames with marker Start and End times for each log
        logs (list of pd.DataFrame): List of log DataFrames with 'timestamps' column

    Returns:
        list of pd.DataFrame: DataFrames indicating clot presence for each log at each time step
    """
    clot_timeseries = []

    for i, log in enumerate(logs):
        # Initialize binary array for this log
        log_clot_timeseries = np.zeros(len(log))
        # Get label times for this log
        start_time = labels[i]['Start'].values[0]
        end_time = labels[i]['End'].values[0]

        if start_time >= end_time:
            start_time, end_time = end_time, start_time

        

        # Set binary values for time steps within the clot
        log_clot_timeseries[(log["timestamps"] >= start_time) & (log["timestamps"] <= end_time)] = 1

        clot_timeseries.append(pd.DataFrame({
            "timestamps": log["timestamps"],
            "in_clot": log_clot_timeseries
        }))

    return clot_timeseries

def reset_timestamps(df, fs=1000):
    """
    Reset timestamps in DataFrame to start from zero with uniform spacing based on sampling frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'timestamps' column
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    pd.DataFrame
        DataFrame with reset timestamps
    """
    df = df.copy()
    dt = 1.0 / fs
    df["timestamps"] = np.arange(len(df)) * dt
    return df


def load_and_preprocess_images(img_paths, rotate_code=None, fx=0.25, fy=0.25):
    """Load, rotate, and downsample grayscale images from a list of paths."""
    imgs = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return np.stack(imgs).astype(np.float32)

def compute_foreground_masks(images, alpha=0.3, threshold=0.05):
    """Compute EMA-based foreground masks for a stack of images."""
    background = images[0]
    masks = []
    for i in range(images.shape[0]):
        diff = np.abs(images[i] - background)
        mask = (diff > threshold).astype(np.float32)
        masks.append(mask)
        background = alpha * images[i] + (1 - alpha) * background
    return masks

def get_cropping_bounds(masks, margin=20):
    """Compute bounding box coordinates covering all masks with margin."""
    full_mask = np.max(np.stack(masks), axis=0)
    ys, xs = np.where(full_mask > 0)
    return (
        max(np.min(xs) - margin, 0), min(np.max(xs) + margin, full_mask.shape[1]),
        max(np.min(ys) - margin, 0), min(np.max(ys) + margin, full_mask.shape[0])
    )

def crop_and_resize_masks(masks, bounds, target_shape=(60, 300)):
    """Crop masks to bounds and resize to target_shape."""
    min_x, max_x, min_y, max_y = bounds
    return [
        cv2.resize(mask[min_y:max_y, min_x:max_x], target_shape[::-1], interpolation=cv2.INTER_AREA)
        for mask in masks
    ]

def preprocess_images(logs, alpha=0.3, threshold=0.05, margin=20, target_shape=(60, 300), save=False):
    """Main preprocessing function for camera masks."""
    all_masks_cam1, all_masks_cam2 = [], []

    cam1_bounds_list, cam2_bounds_list = [], []

    # 1. Load, rotate, downsample, compute masks
    for idx, row in tqdm(logs.iterrows(), total=len(logs), desc="Preprocessing images"):
        cam1_paths, cam2_paths = get_images_paths_from_log(row)
        cam1_imgs = load_and_preprocess_images(cam1_paths, rotate_code=cv2.ROTATE_90_CLOCKWISE)
        cam2_imgs = load_and_preprocess_images(cam2_paths, rotate_code=cv2.ROTATE_180)

        masks_cam1 = compute_foreground_masks(cam1_imgs, alpha, threshold)
        masks_cam2 = compute_foreground_masks(cam2_imgs, alpha, threshold)

        all_masks_cam1.append(masks_cam1)
        all_masks_cam2.append(masks_cam2)

        cam1_bounds_list.append(get_cropping_bounds(masks_cam1, margin))
        cam2_bounds_list.append(get_cropping_bounds(masks_cam2, margin))

    # 2. Compute overall bounds across logs
    overall_bounds_cam1 = (
        min(b[0] for b in cam1_bounds_list), max(b[1] for b in cam1_bounds_list),
        min(b[2] for b in cam1_bounds_list), max(b[3] for b in cam1_bounds_list)
    )
    overall_bounds_cam2 = (
        min(b[0] for b in cam2_bounds_list), max(b[1] for b in cam2_bounds_list),
        min(b[2] for b in cam2_bounds_list), max(b[3] for b in cam2_bounds_list)
    )

    # 3. Crop, resize, and optionally save masks
    for i in tqdm(range(len(all_masks_cam1)), desc="Cropping and resizing masks"):
        all_masks_cam1[i] = crop_and_resize_masks(all_masks_cam1[i], overall_bounds_cam1, target_shape)
        all_masks_cam2[i] = crop_and_resize_masks(all_masks_cam2[i], overall_bounds_cam2, target_shape)

        if save:
            cam1_paths, cam2_paths = get_images_paths_from_log(logs.iloc[i])
            for j, (mask1, mask2) in enumerate(zip(all_masks_cam1[i], all_masks_cam2[i])):
                save_path_1 = str(cam1_paths[j]).replace("raw", "processed").replace(".jpg", "_mask.npz")
                save_path_2 = str(cam2_paths[j]).replace("raw", "processed").replace(".jpg", "_mask.npz")
                os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
                os.makedirs(os.path.dirname(save_path_2), exist_ok=True)
                np.savez_compressed(save_path_1, mask1.astype(np.uint8))
                np.savez_compressed(save_path_2, mask2.astype(np.uint8))

    return all_masks_cam1, all_masks_cam2

if __name__ == "__main__":
    from .load_data import list_logs
    from . import paths 
    heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
    heads_rename = ["timestamps", "force_sensor_v"]
    f_s = 1000
    fss = 568.5
    log_names = list_logs(paths.PAPER_EXPERIMENT_DATA_FOLDER)
    preprocess_images(log_names, save=True)
