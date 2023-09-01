import pickle as pkl
import pandas as pd
import numpy as np
from scipy.stats import norm, kstest, shapiro, anderson
import matplotlib.pyplot as plt
import seaborn as sns


# with open('../pickles/study_histogram.pkl', 'rb') as f:
#     [acute_md_patients,long_md_patients,md_controls] = pkl.load(f)
# f.close()

with open('../pickles/significant_and_corrections/MD_tract_table.pkl', 'rb') as f:
    [data, acute_md_patients, long_md_patients, md_controls] = pkl.load(f)
f.close()
sigs = ['fronto-pontine tract left', 'corticospinal tract left', 'fronto-pontine tract right', 'parieto‐occipital pontine left', 'thalamo-premotor right', 'corticospinal tract right', 'striato-occipital region right', 'arcuate fascicle right', 'striato-premotor region right', 'inferior longitudinal fascicle right', 'thalamo-precentral right', 'middle longitudinal fascicle right', 'cingulum right']


def remove_outliers(df, threshold_z_score):
    """
    Remove rows (patients) from a pandas DataFrame based on a z-score criterion for outlier detection.

    Parameters:
        - df: pandas DataFrame with columns of scan data for different brain regions and rows of patients
        - threshold_z_score: float representing the threshold z-score for outlier detection

    Returns:
        - pandas DataFrame with rows removed for patients with outlier scan data
    """
    # Calculate the z-scores of the data
    z_scores = np.abs((df - df.mean()) / df.std())

    # Create a boolean mask to identify rows with outlier data
    outliers_mask = (z_scores > threshold_z_score).any(axis=1)

    # Filter the DataFrame to only keep rows without outlier data
    df_filtered = df[~outliers_mask]

    return df_filtered

md_controls = remove_outliers(md_controls,2)
acute_md_patients = remove_outliers(acute_md_patients,2)
long_md_patients = remove_outliers(long_md_patients,2)


rois = sigs

def boxer(dfs: list, rois: list, title: str = '',
               x_label: str = '', y_label: str = '', save_name: str = '',
               types: list = ['controls','acute', 'longitudinal']):
    """
    A function that takes a dataframe with columns of scan data and produces a boxplot. The function takes in the
    dataframes associated with different patient types and time nature of the scan. The default is [acute,
    longitudinal, controls] for what these dataframes correspond to. It also takes in a list of relevant regions that
    are to be plotted. It cab also take various inputs to vary the labels on the plot and the savename.
    :param dfs: List of dataframes corresponding to patient or control scan data in the acute or longitudinal timeframe.
    :param rois: List of regions of interests in the brain to be plotted.
    :param types: List detailing what the dataframes correspond to.
    :param title: Title of the plot.
    :param x_label: X label of the plot.
    :param y_label: Y label of the plot.
    :param save_name: Save name for the plot.
    :return: Shows the boxplot and saves it under the save_name accordingly.
    """
    # Checking that a dataframe has been provided for each patient type.
    if len(types) != len(dfs):
        raise Exception('Need same number of patient types as dataframes of patient types.')

    # Make rois input all lowercase
    rois = list(map(lambda x: x.lower(), rois))

    # Remove outliers from the dataframe
    # for i, type_df in enumerate(type_dfs):
    #     type_dfs[i] = outlier_removal(type_df)

    # Create an empty list to store the plot dataframes
    plot_dfs = []

    # We need to flip the table such that we have rows corresponding to each subject and column entries detailing the
    # type of scan and the region of interest, this is necessary for plotting using the seaborn library
    # Loop through each type of data and its corresponding dataframe
    for type, df in zip(types, dfs):
        # Create an empty list to store the dataframes for the same patient type and different regions
        same_type_dfs = []
        # Loop through each region of interest in the dictionary
        for region in rois:
            # Extract the data for the current region from the dataframe and convert it to a list
            temp_df = df[region].tolist()
            # Convert the list to a pandas dataframe with one column named 'value'
            temp_df = pd.DataFrame(temp_df, columns=['value'])
            # Add a column named 'region' to the dataframe and set all rows to the current region
            temp_df.insert(0, 'region', region)
            # Add the new dataframe to the list of dataframes for the same patient type
            same_type_dfs.append(temp_df)
        # Concatenate all dataframes for the same patient type and different regions into a single dataframe
        same_type_df = pd.concat(same_type_dfs)
        # Add a column named 'type' to the dataframe and set all rows to the current patient type
        same_type_df.insert(0, 'type', type)
        # Add the new dataframe to the list of plot dataframes
        plot_dfs.append(same_type_df)

    # Concatenate all plot dataframes into a single dataframe and reset the index
    plot_df = pd.concat(plot_dfs).reset_index()

    sns.set(style='ticks', palette='deep')

    # Increase the size of the figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Create the plot
    sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax)

    # Add a title
    ax.set_title(title, fontsize=20, fontweight='bold')

    # Set the x-axis label
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')

    # Set the y-axis label
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold')

    # Rotate the x-axis labels by 45 degrees
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)

    # Add padding to the x-axis labels
    ax.margins(x=0.1)

    # Set the font size for the tick labels
    ax.tick_params(labelsize=12)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., ncol=1, fontsize='medium', frameon=False,
              title='Patient Type', title_fontsize='large')

    plt.tight_layout()



    plt.savefig( save_name + '.png')
    plt.show()

boxer([md_controls,acute_md_patients,long_md_patients],rois,'Measuring the effect of Covid-19 hospitalisation on mean diffusivity in white matter Tractseg regions', 'White Matter ROI', 'Mean Diffusivity $\mathregular{/mm^{2}s^{-1}}$','histo')

