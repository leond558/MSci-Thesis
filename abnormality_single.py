import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the Gaussian mixture model function with three Gaussian components
def gaussian_mixture(x, A1, A2, A3, X1, X2, X3, s1, s2, s3):
    term1 = A1 * np.exp(-(x - X1) ** 2 / (2 * s1 ** 2))
    term2 = A2 * np.exp(-(x - X2) ** 2 / (2 * s2 ** 2))
    term3 = A3 * np.exp(-(x - X3) ** 2 / (2 * s3 ** 2))
    return term1 + term2 + term3


# Define a single Gaussian function
def gaussian(x, A1, X, s):
    return A1 * np.exp(-(x - X) ** 2 / (2 * s ** 2))


# Main function to fit a Gaussian mixture model to the data
def fit_gaussian_mixture(filename, min_dif=0, max_dif=np.inf):
    # Load data from the file
    data = load_files(filename)
    # Perform the Gaussian mixture fitting and obtain the optimized parameters
    params, xdata, pbin = do_figure(data, min_dif, max_dif)
    # Save the probability maps to separate NIfTI files
    make_maps(filename, params, data)

    return params, xdata, pbin


# Load the image data from the input NIfTI file
def load_files(filename):
    image = nib.load(filename)
    data = image.get_fdata()

    return data


# Fit the Gaussian mixture model to the histogram of the data
def do_figure(p, min_dif, max_dif):
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
    base_path, full_name = os.path.split(filename)
    name, ext = os.path.splitext(full_name)

    return base_path, name, ext

    # Main script


if __name__ == "__main__":
    # Example usage: Replace 'input_file.nii.gz' with your input NIfTI file
    input_file = 'scans/patients/30293.nii.gz'
    min_dif = 0
    max_dif = np.inf

    # Fit the Gaussian mixture model and obtain the parameters, xdata, and pbin
    params, xdata, pbin = fit_gaussian_mixture(input_file, min_dif=min_dif, max_dif=max_dif)
    print("Parameters: ", params)
    plt.savefig('fitplots/fit.png')
    plt.show()

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
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram with Fitted Gaussian Function')

    # Show the plot
    plt.savefig('fitplots/fit.png')
    plt.show()

