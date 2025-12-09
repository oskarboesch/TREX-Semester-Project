import torch
from torch.utils.data import Dataset
from data.load_data import load_data, load_labels
from data.preprocess_data import preprocess_logs, get_label_timeseries, get_images_paths_from_log, get_frame_id_from_path
from sklearn.neighbors import KernelDensity
import numpy as np
from pathlib import Path


class SensorDataset(Dataset):
    def __init__(self, log_names, features=('signal',), fss=568.5, mode='train', downsampling_freq=20, with_augmentation=True, with_kde_weighting=False, mean_force=None, std_force=None):
        self.mode = mode
        self.lengths = []
        self.sample_weights = None
        self.features = features
        self.with_augmentation = with_augmentation
        self.check_features()

        heads_keep = ["timestamp [s]", "Force sensor voltage [V]", "Camera trigger signal"]
        heads_rename = ["timestamps", "force_sensor_v", "camera_trigger_signal"]

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
        self.register_mean_and_std(data_fit, mean_force=mean_force, std_force=std_force)


        if with_kde_weighting:
            self.register_weights(labels)

        for i, df, label in zip(range(len(data_fit)), data_fit, label_timeseries):
            self.add_sample(i, data=df, label=label, log_name=log_names.iloc[i], data_raw=data[i])
            
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
    
    def check_features(self):
        if 'signal' not in self.features and 'bandpower' not in self.features and 'images' not in self.features:
            raise ValueError("At least one of 'signal', 'bandpower' or 'images' must be in features")
        
    def register_mean_and_std(self, data, mean_force=None, std_force=None):
        if mean_force is not None and std_force is not None:
            self.mean_force = mean_force
            self.std_force = std_force
        else:
            if self.mode=='eval':
                # compute mean and std from training set only
                raise ValueError("mean_force and std_force must be provided for eval mode")
            self.mean_force = np.mean([df['force_sensor_mN'].mean() for df in data])
            self.std_force = np.mean([df['force_sensor_mN'].std() for df in data])

    def register_weights(self, labels):
        if self.mode=='eval':
            raise ValueError("kde weighting is only used in training mode")
        # compute sample weights from training set only

        for label in labels:
            length = label['End'].values - label['Start'].values
            self.lengths.extend(length.tolist())
        self.kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(np.array(self.lengths).reshape(-1, 1))
        self.sample_weights = []
        for length in self.lengths:
            log_dens = self.kde.score_samples(np.array(length).reshape(1,-1))
            weight = 1.0 / np.exp(log_dens)
            self.sample_weights.extend(weight.tolist())

        # normalize weights
        self.sample_weights = np.array(self.sample_weights)
        self.sample_weights = self.sample_weights / np.mean(self.sample_weights)
        self.sample_weights = self.sample_weights ** 2  # amplify differences
        
    def add_sample(self, i, data, label,log_name, data_raw):
        x_list = []
        masks = None
        label = label['in_clot'].values          # binary sequence

        if 'signal' in self.features:
            self.add_signal_data(x_list, data)


        if 'bandpower' in self.features:
            self.add_frequency_data(x_list, data)

        # concatenate signal and bandpower data if any
        x = np.concatenate(x_list, axis=1) if len(x_list) > 0 else None

        if 'images' in self.features:
            masks = self.get_masks(data, data_raw, log_name)

        # add preprocessed signal, freq and image data 
        self.samples.append((x, masks, label.reshape(-1,1)))
    

        if self.with_augmentation and self.mode == 'train' and x is not None:
            self.augment_sample(i, x, masks, label)



    def add_signal_data(self, x_list, data):
        force = data[['force_sensor_mN']].values  # pick your sensor columns
        # Normalize forces
        force = (force - self.mean_force) / self.std_force
        x_list.append(force)

    def add_frequency_data(self, x_list, data):
        bandpower_cols = [col for col in data.columns if col.startswith('band_power_')]
        x_bandpower = data[bandpower_cols].values
        x_list.append(x_bandpower)

    def get_masks(self, data, data_raw, log_name):
        img_1_paths, img_2_paths = get_images_paths_from_log(log_name, processed=True)

        # STEP 1 — Load masks and store keyed by frame_id
        masks_dict_1 = {}
        masks_dict_2 = {}
        frame_ids_1 = []
        frame_ids_2 = []

        for p1, p2 in zip(img_1_paths, img_2_paths):

            data_1 = np.load(p1)
            data_2 = np.load(p2)
            path_1 = Path(str(data_1["frame_id"]))
            path_2 = Path(str(data_2["frame_id"]))
            # check if path_1 is an int
            if path_1.name.isdigit():
                fid1 = int(path_1.name)
                fid2 = int(path_2.name)
            else:
                fid1 = int(get_frame_id_from_path(path_1.name))
                fid2 = int(get_frame_id_from_path(path_2.name))

            frame_ids_1.append(fid1)
            frame_ids_2.append(fid2)

            masks_dict_1[fid1] = data_1['mask'][None, :, :]
            masks_dict_2[fid2] = data_2['mask'][None, :, :]

        # STEP 2 — Keep only the frames present in BOTH cameras
        frame_ids_common = sorted(set(frame_ids_1) & set(frame_ids_2))

        # STEP 3 — Build new masks_list ONLY from common frame IDs
        masks_list = []
        trigger_signal = data_raw['camera_trigger_signal'].values
        # obtain trigger timestamps where trigger signal goes from 0 to 1
        trigger_indices = np.where((trigger_signal[:-1] == 0) & (trigger_signal[1:] == 1))[0] + 1
        if max(frame_ids_common) >= len(trigger_indices):
            print(f"Warning: More frame indices than triggers indices. Cropping")
            # get rid of values higher than the number of trigger indices
            frame_ids_common = [fid for fid in frame_ids_common if fid < len(trigger_indices)]
        # take only indices at frame_ids_common

        frame_id_to_trigger_index = {
            frame_id: trigger_indices[frame_id]
            for frame_id in frame_ids_common
        }

        trigger_timestamps = np.array([
            data_raw['timestamps'].values[frame_id_to_trigger_index[fid]]
            for fid in frame_ids_common
        ])

        for fid in frame_ids_common:
            mask1 = masks_dict_1[fid]
            mask2 = masks_dict_2[fid]
            masks_list.append(np.concatenate([mask1, mask2], axis=0))


        # STEP 4 — Align to x_timestamps
        x_timestamps = data['timestamps'].values

        # restrict mask timestamps to >= first x timestamp
        valid = trigger_timestamps >= x_timestamps[0]
        trigger_timestamps = trigger_timestamps[valid]
        masks_list = [m for (m, v) in zip(masks_list, valid) if v]

        # STEP 5 — Nearest-neighbor alignment
        masks_aligned = []
        for t in x_timestamps:
            idx = np.argmin(np.abs(trigger_timestamps - t))
            masks_aligned.append(masks_list[idx])

        masks = np.array(masks_aligned)
        return masks

    def augment_sample(self, i, x, masks, label):
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
