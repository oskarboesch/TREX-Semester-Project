BUCKLING_PATH_TO_RESULTS = "../results/Buckling/2025-11-10_09-12-23"
DOUBLESLOPE_PATH_TO_RESULTS = "../results/DoubleSlope/2025-11-10_09-12-44"

from pathlib import Path   
import numpy as np

# for all files in subfolders in BUCKLING_PATH_TO_RESULTS
buckling_start_accuracies = []
buckling_end_accuracies = []
for path in Path(BUCKLING_PATH_TO_RESULTS).rglob("buckling_model*"):
    buckling_result = np.load(path, allow_pickle=True)
    buckling_start_accuracies.append(buckling_result['start_accuracy'])
    buckling_end_accuracies.append(buckling_result['end_accuracy'])

print(f"Buckling Model - Start Accuracy: Mean = {np.mean(buckling_start_accuracies):.4e}, Std = {np.std(buckling_start_accuracies):.4e}")
print(f"Buckling Model - End Accuracy: Mean = {np.mean(buckling_end_accuracies):.4e}, Std = {np.std(buckling_end_accuracies):.4e}")


double_slope_start_accuracies = []
double_slope_end_accuracies = []
for path in Path(DOUBLESLOPE_PATH_TO_RESULTS).rglob("doubleslope_model*"):
    doubleslope_result = np.load(path, allow_pickle=True)
    double_slope_start_accuracies.append(doubleslope_result['start_accuracy'])
    double_slope_end_accuracies.append(doubleslope_result['end_accuracy'])

print(f"Double Slope Model - Start Accuracy: Mean = {np.mean(double_slope_start_accuracies):.4e}, Std = {np.std(double_slope_start_accuracies):.4e}")
print(f"Double Slope Model - End Accuracy: Mean = {np.mean(double_slope_end_accuracies):.4e}, Std = {np.std(double_slope_end_accuracies):.4e}")

# plot boxplots of start and end accuracies for both models
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].boxplot([buckling_start_accuracies, double_slope_start_accuracies], tick_labels=['Buckling', 'Double Slope'])
axs[0].set_title('Start Accuracies')
axs[0].set_ylabel('Accuracy')
axs[1].boxplot([buckling_end_accuracies, double_slope_end_accuracies], tick_labels=['Buckling', 'Double Slope'])
axs[1].set_title('End Accuracies')
plt.tight_layout()
plt.savefig('model_accuracies_boxplot.png')
plt.show()