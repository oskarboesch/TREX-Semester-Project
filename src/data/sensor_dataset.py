import torch
from torch.utils.data import Dataset
from data.load_data import load_data, load_labels
from data.preprocess_data import preprocess_logs, get_label_timeseries, get_images_paths_from_log
from sklearn.neighbors import KernelDensity
import numpy as np


class SensorDataset(Dataset):
    def __init__(self, log_names, features=('signal',), fss=568.5, mode='train', downsampling_freq=20, with_augmentation=True, with_kde_weighting=False, mean_force=None, std_force=None):
        self.mode = mode
        self.lengths = []
        self.sample_weights = None
        self.features = features
        if 'signal' not in features and 'bandpower' not in features and 'images' not in features:
            raise ValueError("At least one of 'signal', 'bandpower' or 'images' must be in features")

        heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
        heads_rename = ["timestamps", "force_sensor_v"]

        data = load_data(log_names, heads_keep, heads_rename, fss)
        # get direction from first log
        direction = log_names.iloc[0]['direction']
        data_fit = preprocess_logs(data,
                                     head='force_sensor_mN',
                                     downsampling_freq=downsampling_freq,
                                     direction=direction, with_bandpower=('bandpower' in features))
        labels = load_labels(log_names)
        label_timeseries = get_label_timeseries(labels, data_fit)

        self.samples = []
        if mean_force is not None and std_force is not None:
            self.mean_force = mean_force
            self.std_force = std_force
        else:
            if mode=='eval':
                # compute mean and std from training set only
                raise ValueError("mean_force and std_force must be provided for eval mode")
            self.mean_force = np.mean([df['force_sensor_mN'].mean() for df in data_fit])
            self.std_force = np.mean([df['force_sensor_mN'].std() for df in data_fit])

        if with_kde_weighting:
            if mode=='eval':
                raise ValueError("kde weighting is only used in training mode")
            # compute sample weights from training set only
            all_forces = np.concatenate([df[['force_sensor_mN']].values for df in data_fit], axis=0)
            self.kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(all_forces)
            self.sample_weights = []
            for label in labels:
                length = label['End'].values - label['Start'].values
                self.lengths.extend(length.tolist())
                log_dens = self.kde.score_samples(np.array(length).reshape(1,-1))
                weight = 1.0 / np.exp(log_dens)
                self.sample_weights.extend(weight.tolist())

            # normalize weights
            self.sample_weights = np.array(self.sample_weights)
            self.sample_weights = self.sample_weights / np.mean(self.sample_weights)
            self.sample_weights = self.sample_weights ** 2  # amplify differences

        for i, df, label in zip(range(len(data_fit)), data_fit, label_timeseries):
            x_list = []
            masks = None
            label = label['in_clot'].values          # binary sequence


            if 'signal' in self.features:
                force = df[['force_sensor_mN']].values  # pick your sensor columns
                # Normalize forces
                force = (force - self.mean_force) / self.std_force
                x_list.append(force)

            if 'bandpower' in self.features:
                bandpower_cols = [col for col in df.columns if col.startswith('band_power_')]
                x_bandpower = df[bandpower_cols].values
                x_list.append(x_bandpower)
            
            x = np.concatenate(x_list, axis=1) if len(x_list) > 0 else None

            if 'images' in self.features:
                img_1_paths, img_2_paths = get_images_paths_from_log(log_names.iloc[i])
                # processed paths are the same but change raw with processed and jpg with .npz
                masks_list = []
                for p1, p2 in zip(img_1_paths, img_2_paths):
                    proc_p1 = str(p1).replace('raw', 'processed').replace('.jpg', '_mask.npz')
                    proc_p2 = str(p2).replace('raw', 'processed').replace('.jpg', '_mask.npz')
                    mask1 = np.load(proc_p1)['arr_0'][None, :, :]
                    mask2 = np.load(proc_p2)['arr_0'][None, :, :]
                    mask = np.concatenate([mask1, mask2], axis=0)  # concatenate embeddings from both images
                    masks_list.append(mask)
                # images are sampled at 10 Hz
                mask_timestamps = np.arange(len(masks_list)) / 10.0
                # need to align masks to x timestamps
                x_timestamps = df['timestamps'].values
                print(f"Aligning {mask_timestamps} to {x_timestamps}")

                # cut masks to first x_timestamps
                mask_timestamps = mask_timestamps[mask_timestamps >= x_timestamps[0]]
                # interpolate masks to x timestamps closest indices
                masks_aligned = []
                for t in x_timestamps:
                    idx = np.argmin(np.abs(mask_timestamps - t))
                    masks_aligned.append(masks_list[idx])
                masks = np.array(masks_aligned)

            if with_augmentation and mode == 'train' and x is not None:
                x_noise = x + np.random.normal(0, 0.01, x.shape)
                x_scale = x * np.random.uniform(0.95, 1.05, (x.shape[0],1))
                x_noise_scale = x * np.random.uniform(0.95, 1.05, (x.shape[0],1)) + np.random.normal(0, 0.01, x.shape)
                self.samples.append((x_noise, masks, label.reshape(-1,1)))
                self.samples.append((x_scale, masks, label.reshape(-1,1)))
                self.samples.append((x_noise_scale, masks, label.reshape(-1,1)))
                # need to add weights accordingly if using kde weighting
                if self.sample_weights is not None:
                    self.sample_weights = np.append(self.sample_weights, self.sample_weights[i])
                    self.sample_weights = np.append(self.sample_weights, self.sample_weights[i])
                    self.sample_weights = np.append(self.sample_weights, self.sample_weights[i])

            self.samples.append((x, masks, label.reshape(-1,1)))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, masks, y = self.samples[idx]
        w = 1.0
        if self.sample_weights is not None:
            w = self.sample_weights[idx]
        return (
            torch.tensor(x, dtype=torch.float32) if x is not None else torch.empty((0,0), dtype=torch.float32), # return empty tensor if no sensor data
            torch.tensor(masks, dtype=torch.float32) if masks is not None else torch.empty((0,0,0,0), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
        )
    @property
    def input_size(self):
        x0, _, _ = self.samples[0]
        return x0.shape[1] if x0 is not None else 0