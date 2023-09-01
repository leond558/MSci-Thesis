# abnormality_prob_maps
"""This Python script processes and visualizes MRI brain scans to detect and highlight abnormal regions in a
patient's brain compared to a control group. It calculates an abnormality score based on differences in Gaussian
Mixture Model (GMM) components between the patient and control group scans, and generates a set of plots for each GMM
component showing the patient's map, the average control map, and the abnormality score map."""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import matplotlib.colors as colors

# Define function to display abnormality maps
def display_maps(patient_file, control_files, slice_index, min_distance=5, num_peaks=5):
    """
       Function to display probability maps for each Gaussian component for the patient and for the controls
       and the difference between them.

       :param patient_file: str, file path to the patient's .nii.gz files
       :param control_files: list of str, file paths to the control group's .nii.gz files
       :param slice_index: int, index of the slice to be displayed from the 3D volume
       :param min_distance: int, minimum number of pixels separating peaks in the abnormality score map, defaults to 5
       :param num_peaks: int, number of peaks to identify in the abnormality score map, defaults to 5

       :return: None, this function saves figures to the local filesystem and writes information to .txt files

       The function loads Gaussian component maps for the patient and the control group. It then calculates a
       mean control map for each component and computes the absolute difference between the patient's maps and the mean control maps.

       This difference is considered as an abnormality score which is then written into a .txt file. The function also generates a
       set of plots for each Gaussian component showing the patient's map, the average control map, and the abnormality score map.

       The function also identifies the local maxima in the abnormality score map, which are marked on the plot and saved into a .txt file.
       """

    # Lists to store patient and control maps
    patient_maps = []
    control_maps = []

    # List to store coordinates of maximum values
    maxima_coordinates = []

    # Loop over the three components of the .nii files
    for i in range(1, 4):
        # Load patient map data
        patient_map_file = patient_file + f".nii_component_{i}.nii.gz"
        patient_map = nib.load(patient_map_file).get_fdata()
        patient_maps.append(patient_map)

        # Initialize an empty control map
        control_map_sum = np.zeros_like(patient_map)

        # Load each control map data and add to the sum
        for control_file in control_files:
            control_map_file = control_file + f".nii_component_{i}.nii.gz"
            control_map = nib.load(control_map_file).get_fdata()
            control_map_sum += control_map

        # Calculate the average control map and append to control_maps
        control_map_avg = control_map_sum / len(control_files)
        control_maps.append(control_map_avg)

    # Calculate the average control map for all components
    avg_control_maps = np.mean(control_maps, axis=0)

    # Calculate the difference between patient maps and average control maps
    diff_maps = np.abs(patient_maps - avg_control_maps)

    # Calculate a total abnormality score for the patient
    total_abnormality_score = np.mean(diff_maps)

    # Write the total abnormality score to a text file
    with open('FAabnormalities/' + patient_file[-7:] + '.txt', mode='w') as file:
            file.write(str(total_abnormality_score))

    # Create subplots to visualize the maps
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Probability maps for each Gaussian component \n for the patient and for the controls\n and the difference between them.', fontsize=16)

    # Add overall labels for all subplots
    fig.text(0.5, 0.04, 'Spatial Coordinate X of Scan Image', ha='center', va='center', fontsize=12)
    fig.text(0.04, 0.5, 'Spatial Coordinate Y of Scan Image', ha='center', va='center', rotation='vertical', fontsize=12)

    # Normalize the colormap so it's consistent across all components
    diff_map_norm = colors.Normalize(vmin=np.min(diff_maps), vmax=np.max(diff_maps))

    # Loop over the three components to plot patient maps, control maps, and abnormality score maps
    for i in range(3):
        axs[i, 0].imshow(patient_maps[i][:, :, slice_index], cmap='gray', origin='lower')
        axs[i, 0].set_title(f'Patient - Gaussian Component {i + 1}')
        axs[i, 1].imshow(control_maps[i][:, :, slice_index], cmap='gray', origin='lower')
        axs[i, 1].set_title(f'Average Control - Gaussian Component {i + 1}')
        diff_map = np.abs(patient_maps[i][:, :, slice_index] - control_maps[i][:, :, slice_index])
        im = axs[i, 2].imshow(diff_map, cmap='hot', norm=diff_map_norm, origin='lower')
        axs[i, 2].set_title(f'Abnormality Score - Gaussian Component {i + 1}')

    for ax in axs.flat:
        ax.label_outer()

    # Find the local maxima in the abnormality score map
    coordinates = peak_local_max(diff_map, min_distance=min_distance, num_peaks=num_peaks)
    maxima_coordinates.append(coordinates)

    # Mark the local maxima on the abnormality score map
    for coord in coordinates:
        axs[i, 2].plot(coord[1], coord[0], 'bo', markersize=5, markeredgewidth=1, markerfacecolor='None')

    # Add a colorbar showing the scale for the colormap
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Abnormality Score Colourbar', rotation=270, labelpad=20)

    # Save the plot
    plt.savefig('FAplots/'+str(slice_index) + '_' + patient_file[-7:])

    # Save the outputs of the most abnormal regions
    with open('FAoutputs/' +str(slice_index) + '_' + patient_file[-7:] + '.txt', mode='w') as file:
        for i, coords in enumerate(maxima_coordinates):
            file.write(f"Component {i+1} local maxima coordinates:")
            for coord in coords:
                file.write(f"  - x: {coord[1]}, y: {coord[0]}")

# Example usage
if __name__ == '__main__':
    # Define the patient and control file pathnames
    patient_files = ['30292', '30293', '30295', '30297', '30302']
    control_files = ['10000', '10058', '10179', '10459', '14136', '16242', '18770', '21091', '23108', '24972', '25002',
                     '25044', '27336', '28069', '28973']

    patient_files = list(map(lambda filename: 'patients_FA_scans/' + filename +'FA', patient_files))
    control_files = list(map(lambda filename: 'controls_FA_scans/' + filename +'FA', control_files))

    # Define a slice to be used for analysis
    slice_index = 50

    # Run the function to produce the abnormality score map plots
    for patient_file in patient_files:
        display_maps(patient_file, control_files, slice_index)
