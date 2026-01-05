import time
import torch
import yaml

import data.load_data as ld
from data.sensor_dataset import SensorDataset
from models.gru import load_gru_model
from sklearn.model_selection import train_test_split
from utils.seed import set_seed

model_path = "../models/gru_model_2025-12-15_12-13-28.pt"
cfg_path = "../results/gru_results/2025-12-15_12-13-28/complete_config.yml"

# Load config
with open(cfg_path, 'r') as f:
    config = yaml.safe_load(f)

# Setup device (GPU only)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    raise RuntimeError("GPU not available. This script requires CUDA.")
print(f"Using device: {device}")

# Load model
input_size = 1
model = load_gru_model(model_path, input_size, config, device)

# Prepare test data
set_seed(42)
data, _ = ld.get_extra_logs()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_dataset = SensorDataset(log_names=test_data, mode='eval', with_augmentation=False, features=('signal',), downsampling_freq=10, mean_force=0, std_force=1)

# Warmup
print("\nWarming up GPU...")
for _ in range(10):
    warmup_sample = test_dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(warmup_sample)

# Measure inference time
print(f"\nMeasuring inference time on {len(test_dataset)} samples...")
torch.cuda.synchronize()
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    start_time.record()
    for i in range(len(test_dataset)):
        input_sample = test_dataset[i][0].unsqueeze(0).to(device)
        _ = model(input_sample)
    end_time.record()

torch.cuda.synchronize()
total_time_ms = start_time.elapsed_time(end_time)
avg_inference_time = total_time_ms / len(test_dataset)

print(f"\n{'='*50}")
print(f"Inference Time Results:")
print(f"{'='*50}")
print(f"Total inference time for {len(test_dataset)} samples: {total_time_ms:.2f} ms")
print(f"Average inference time per sample: {avg_inference_time:.4f} ms")
print(f"Throughput: {1000/avg_inference_time:.2f} samples/sec")
print(f"{'='*50}")
