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

def downsample(df, factor):
    # Downsample every 'factor' points
    df = df.copy()
    return df.iloc[::factor, :].reset_index(drop=True)

def preprocess_logs(logs_fit, data_fit, data_fit_plot):
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
