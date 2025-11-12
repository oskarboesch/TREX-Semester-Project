import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import sosfreqz, butter, zpk2sos, sosfiltfilt

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
    n_downsample = round(f_s * 0.2)
    head = "force_sensor_mN"

    # Design filter
    sos, _ = design_butter(f_s, 10, 3, "low")

    # Start preprocessing
    start_time = time.time()

    for i, df in tqdm(enumerate(data_fit)):
        ydata = df[head].values
        ydata = ydata - ydata[0]  # Remove initial value to avoid step
        ydata = sosfiltfilt(sos, ydata)
        df[head] = ydata
        # Crop bounds
        direction = logs_fit.loc[i, "direction"].lower()
        if direction == "forward":
            range_crop = [5, 0]
        else:
            range_crop = [2, 0]
        df_cropped = crop_data(df, df["timestamps"], range_crop)
        data_fit_plot[i] = df_cropped.copy()  # Not downsampled

        # Downsample
        df_down = downsample(df_cropped, n_downsample)

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

def preprocess_log(df, head, direction, sampling_rate=1000, filter_order=3, downsampling_freq=10):
    df = df.copy()
    lowpass_cutoff = downsampling_freq / 2.0  # Nyquist frequency after downsampling
    sos, _ = design_butter(sampling_rate, lowpass_cutoff, filter_order, 'low')
    data = df[head].values
    data = data - data[0]  # Remove initial value to avoid step
    data = sosfiltfilt(sos, data)

    # Crop bounds
    if direction == "Forward":
        range_crop = [5, 0]
    else:
        range_crop = [2, 0]
    df_cropped = crop_data(df, df["timestamps"], range_crop)
    # Downsample
    df_down = downsample(df_cropped, sampling_rate, downsampling_freq)

    # Baseline correction
    baseline = df_down[head].iloc[0]
    df_down[head] = df_down[head] - baseline
    df_cropped[head] = df_cropped[head] - baseline

    return df_down

def preprocess_logs(df_list, head, direction, sampling_rate=1000, filter_order=3, downsampling_freq=10):
    processed_list = []
    for df in df_list:
        processed = preprocess_log(df, head, direction, sampling_rate, filter_order, downsampling_freq)
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
