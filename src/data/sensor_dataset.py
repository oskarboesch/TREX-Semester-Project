import torch
from torch.utils.data import Dataset
from data.load_data import load_data, load_labels
from data.preprocess_data import preprocess_logs, get_label_timeseries


class SensorDataset(Dataset):
    def __init__(self, log_names, window_size=1, fss=568.5, mode='train', downsampling_freq=1):
        """
        mode: 'train' -> sliding windows
              'eval'  -> full curve
        """
        self.mode = mode
        self.window_size = window_size

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

        for df, label in zip(data_fit, label_timeseries):
            forces = df[['force_sensor_mN']].values  # pick your sensor columns
            label = label['in_clot'].values          # binary sequence

            # normalize per-curve
            forces = (forces - forces.mean(axis=0)) / (forces.std(axis=0) + 1e-8)

            if mode == 'train':
                # sliding windows
                for i in range(len(forces) - window_size):
                    x = forces[i:i+window_size]
                    y = label[i:i+window_size].reshape(-1,1)
                    self.samples.append((x, y))
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



