# abnormality_multpiple.py
"""This script implements a pipeline for processing MRI scan data to identify abnormalities by fitting a Gaussian
mixture model. It starts by loading MRI scan data from NIfTI files. Then, it computes a histogram of the data and
fits a Gaussian mixture model with three components to this histogram. The optimized parameters of the Gaussian
mixture model are used to create probability maps for each Gaussian component. These probability maps are saved as
separate NIfTI files for further analysis. The script also includes functionality for visualizing the histogram of
the MRI data and the fitted Gaussian mixture model, as well as the individual Gaussian components. This script
supports parallel processing of multiple files for increased efficiency. The results of the Gaussian mixture model
fitting are logged and can be used for further statistical analysis and comparison between different groups of scans
(e.g., patient vs. control)."""

import logging
import multiprocessing
import nibabel as nib
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian mixture model function with three Gaussian components
def gaussian_mixture(x, A1, A2, A3, X1, X2, X3, s1, s2, s3):
    """
    Computes the value of a Gaussian mixture model with three components at given points.

    :param x: Array of points at which to evaluate the Gaussian mixture model.
    :param A1: Amplitude of the first Gaussian component.
    :param A2: Amplitude of the second Gaussian component.
    :param A3: Amplitude of the third Gaussian component.
    :param X1: Mean of the first Gaussian component.
    :param X2: Mean of the second Gaussian component.
    :param X3: Mean of the third Gaussian component.
    :param s1: Standard deviation of the first Gaussian component.
    :param s2: Standard deviation of the second Gaussian component.
    :param s3: Standard deviation of the third Gaussian component.
    :return: Array of Gaussian mixture model values at the points in `x`.
    """
    term1 = A1 * np.exp(-(x - X1) ** 2 / (2 * s1 ** 2))
    term2 = A2 * np.exp(-(x - X2) ** 2 / (2 * s2 ** 2))
    term3 = A3 * np.exp(-(x - X3) ** 2 / (2 * s3 ** 2))
    return term1 + term2 + term3


# Define a single Gaussian function
def gaussian(x, A1, X, s):
    """
    Computes the value of a Gaussian function at given points.

    :param x: Array of points at which to evaluate the Gaussian function.
    :param A1: Amplitude of the Gaussian.
    :param X: Mean of the Gaussian.
    :param s: Standard deviation of the Gaussian.
    :return: Array of Gaussian function values at the points in `x`.
    """
    return A1 * np.exp(-(x - X) ** 2 / (2 * s ** 2))


# Main function to fit a Gaussian mixture model to the data
def fit_gaussian_mixture(filename):
    """
    Fits a Gaussian mixture model to MRI scan data.

    :param filename: Name of the file containing MRI scan data.
    :return: Tuple containing the optimized parameters, the data points, and the corresponding histogram.
    """
    # Define expected minimum and maximum values
    min_dif = 0
    max_dif = np.inf
    # Load data from the file
    data = load_files(filename)
    # Perform the Gaussian mixture fitting and obtain the optimized parameters
    params, xdata, pbin = do_figure(data, min_dif, max_dif)
    # Save the probability maps to separate NIfTI files
    make_maps(filename, params, data)

    return params, xdata, pbin


# Load the image data from the input NIfTI file
def load_files(filename):
    """
    Loads MRI scan data from a NIfTI file.

    :param filename: Name of the NIfTI file.
    :return: Array containing the MRI scan data.
    """
    image = nib.load(filename)
    data = image.get_fdata()

    return data


# Fit the Gaussian mixture model to the histogram of the data
def do_figure(p, min_dif, max_dif):
    """
    Fits a Gaussian mixture model to the histogram of MRI scan data and plots the results.

    :param p: Array containing the MRI scan data.
    :param min_dif: Minimum expected value.
    :param max_dif: Maximum expected value.
    :return: Tuple containing the optimized parameters, the data points, and the normalized histogram.
    """
    # Calculate the histogram of the data
    pbin, edges = np.histogram(p.ravel(), bins=500)
    xdata = (edges[:-1] + edges[1:]) / 2

    # Normalize the histogram
    num_voxels = np.sum(pbin)
    pbin = pbin / num_voxels

    # Set initial parameter estimates
    X0 = np.zeros(9)

    bounds = (np.zeros(9), np.ones(9))

    X0[0] = np.max(pbin)
    X0[1] = np.max(pbin)
    X0[2] = np.max(pbin)
    X0[3] = np.sum(pbin * xdata) / np.sum(pbin)
    X0[4] = 1.1 * X0[3]
    X0[5] = 1.2 * X0[3]
    Xsq = np.sum(pbin * xdata * xdata) / np.sum(pbin)
    X0[6] = np.sqrt(Xsq - (X0[3] * X0[3]))
    X0[7] = X0[6]
    X0[8] = X0[6]

    # Perform the curve fitting using the Gaussian mixture model function
    print('Working')
    params, _ = curve_fit(gaussian_mixture, xdata, pbin, p0=X0, bounds=bounds, max_nfev=100000)

    # Plot the original histogram data in red
    plt.plot(xdata, pbin, 'r')

    # Plot the fitted Gaussian mixture model in green
    ydata = gaussian_mixture(xdata, *params)
    plt.plot(xdata, ydata, 'g')

    # Set the plot labels and title
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Data Distribution (Red) and Model Fit (Green)')
    plt.legend(['Data', 'Fit'])

    return params, xdata, pbin


# Create probability maps and save them as separate NIfTI files
def make_maps(input_file, params, pin):
    """
    Creates probability maps for each Gaussian component and saves them as separate NIfTI files.

    :param input_file: Name of the file containing the original MRI scan data.
    :param params: Optimized parameters of the Gaussian mixture model.
    :param pin: Array containing the MRI scan data.
    """
    input_img = nib.load(input_file)
    base_path, name, _ = fileparts(input_file)

    # Initialize probability map array
    prob = np.zeros((*input_img.shape, 3))

    # Calculate probability maps for each Gaussian component
    for i in range(3):
        prob[..., i] = params[i] * np.exp(-((pin - params[i + 3]) / params[i + 6]) ** 2)

        # Normalize the probability maps
    sump = np.finfo(float).eps + np.sum(prob, axis=-1)

    for i in range(3):
        prob[..., i] = (prob[..., i] / sump)

    # Save the probability maps as separate NIfTI files
    for i in range(3):
        out_nii = nib.Nifti1Image(prob[..., i], input_img.affine, input_img.header)
        nib.save(out_nii, os.path.join(base_path, f"{name}_component_{i + 1}.nii.gz"))

    # Extract the file path, name, and extension from the input filename


def fileparts(filename):
    """
    Extracts the file path, name, and extension from a filename.

    :param filename: Name of the file.
    :return: Tuple containing the file path, name, and extension.
    """
    base_path, full_name = os.path.split(filename)
    name, ext = os.path.splitext(full_name)

    return base_path, name, ext


def combination_plotter(params, xdata, pbin):
    """
    Plots the histogram of MRI scan data and the fitted Gaussian mixture model, as well as the individual Gaussian components.

    :param params: Optimized parameters of the Gaussian mixture model.
    :param xdata: Array of data points.
    :param pbin: Corresponding histogram.
    """
    # Plot the histogram
    plt.bar(xdata, pbin, width=np.diff(xdata)[0], alpha=0.5, label='Histogram')

    # Plot the fitted Gaussian function
    ydata_fitted = gaussian_mixture(xdata, *params)
    plt.plot(xdata, ydata_fitted, 'r-', label='Fitted Gaussian')

    # Plot individual Gaussian peaks
    gaussian1 = gaussian(xdata, A1=params[0], X=params[3], s=params[6])
    gaussian2 = gaussian(xdata, A1=params[1], X=params[4], s=params[7])
    gaussian3 = gaussian(xdata, A1=params[2], X=params[5], s=params[8])

    plt.plot(xdata, gaussian1, 'g--', label='Gaussian 1')
    plt.plot(xdata, gaussian2, 'b--', label='Gaussian 2')
    plt.plot(xdata, gaussian3, 'm--', label='Gaussian 3')

    # Add legend and labels
    plt.legend()
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Histogram with Fitted Gaussian Function')

    # Show the plot
    plt.show()


def process_group(file_list):
    """
    Fits a Gaussian mixture model to the MRI scan data in each file of a list in parallel, handling any fitting errors.

    :param file_list: List of file names containing MRI scan data.
    :return: List of results for each file
    """
    # Use parallelisation to speed up execution time
    with multiprocessing.Pool() as pool:
        group_results = []
        for file in file_list:
            try:
                result = pool.apply_async(fit_gaussian_mixture, [file]).get()
                group_results.append(result[0])
            except RuntimeError:
                # If the fitting fails, log the failed file and move on
                logging.exception(f"Failed to fit {file}. Skipping...")
    return group_results


# Main script
if __name__ == "__main__":
    # Define the patient and control file pathnames
    patient_files = ['30292', '30293', '30294', '30295', '30297', '30302']
    control_files = ['10000', '10058', '10179', '10459', '14136', '16242', '18770', '21091', '23108', '24972', '25002',
                     '25044', '27336', '27949', '28069', '28973']

    patient_files = list(map(lambda filename: 'patients_FA_scans/' + filename + 'FA.nii.gz', patient_files))
    control_files = list(map(lambda filename: 'controls_FA_scans/' + filename + 'FA.nii.gz', control_files))

    patient_results = process_group(patient_files)
    control_results = process_group(control_files)

    # Store the Gaussian Components in a pickle file

    with open('results.pkl', 'wb') as f:
        pkl.dump(
            [control_results, patient_results], f
        )
    f.close()
