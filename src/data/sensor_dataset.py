import torch
from torch.utils.data import Dataset
from data.load_data import load_data, load_labels
from data.preprocess_data import preprocess_logs, get_label_timeseries
import numpy as np


class SensorDataset(Dataset):
    def __init__(self, log_names, window_size=1, fss=568.5, mode='train', downsampling_freq=1, with_image=False, with_augmentation=True, mean_force=None, std_force=None):
        """
        mode: 'train' -> sliding windows
              'eval'  -> full curve
        """
        self.mode = mode
        self.window_size = int(window_size * downsampling_freq)  # convert to number of samples

        heads_keep = ["timestamp [s]", "Force sensor voltage [V]"]
        heads_rename = ["timestamps", "force_sensor_v"]

        data = load_data(log_names, heads_keep, heads_rename, fss)
        # get direction from first log
        direction = log_names.iloc[0]['direction']
        data_fit = preprocess_logs(data,
                                     head='force_sensor_mN',
                                     downsampling_freq=downsampling_freq,
                                     direction=direction)
        labels = load_labels(log_names)
        label_timeseries = get_label_timeseries(labels, data_fit)

        self.samples = []
        if mean_force is not None and std_force is not None:
            self.mean_force = mean_force
            self.std_force = std_force
        else:
            self.mean_force = np.mean([df['force_sensor_mN'].mean() for df in data_fit])
            self.std_force = np.mean([df['force_sensor_mN'].std() for df in data_fit])

        for df, label in zip(data_fit, label_timeseries):
            forces = df[['force_sensor_mN']].values  # pick your sensor columns
            label = label['in_clot'].values          # binary sequence
            # Normalize forces
            forces = (forces - self.mean_force) / self.std_force


            if mode == 'train':
                for i in range(len(forces) - window_size):
                    x = forces[i:i+window_size].copy()
                    y = label[i:i+window_size].reshape(-1,1)

                    # Original window
                    self.samples.append((x, y))
                    if with_augmentation:
                        # Augmentation 1: noise
                        x_aug = x + np.random.normal(0, 0.01, x.shape)
                        self.samples.append((x_aug, y))

                        # Augmentation 2: scaling
                        x_aug = x * np.random.uniform(0.95, 1.05)
                        self.samples.append((x_aug, y))

                        # Augmentation 3: noise + scaling
                        x_aug = x * np.random.uniform(0.95, 1.05) + np.random.normal(0, 0.01, x.shape)
                        self.samples.append((x_aug, y))
            elif mode == 'eval':
                # full curve
                self.samples.append((forces, label.reshape(-1,1)))
            else:
                raise ValueError("mode must be 'train' or 'eval'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
