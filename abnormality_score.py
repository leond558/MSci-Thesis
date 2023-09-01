import numpy as np
import pickle as pkl

with open('results.pkl', 'rb') as f:
    [control_results,patient_results] = pkl.load(f)
f.close()

def calculate_mean_std(group_results):
    group_results_np = np.array(group_results)
    mean_params = np.mean(group_results_np, axis=0)
    std_params = np.std(group_results_np, axis=0)
    return mean_params, std_params

def calculate_z_scores(patient_results, mean_params, std_params):
    patient_results_np = np.array(patient_results)
    z_scores = (patient_results_np - mean_params) / std_params
    return z_scores

mean_control_params, std_control_params = calculate_mean_std(control_results)
z_scores = calculate_z_scores(patient_results, mean_control_params, std_control_params)

# Define a threshold for abnormality (e.g., 1.96 for a 95% confidence interval)
threshold = 1.96

# Find the abnormal voxels
abnormal_voxels = np.abs(z_scores) > threshold

print("Z-scores:", z_scores)
print("Abnormal voxels:", abnormal_voxels)
