# Importing necessary libraries
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# Importing custom functions from the module 'sheet_functions'
from sheet_functions import multiple_scans, extreme_dataframes, group_metric_split, outlier_removal

if __name__ == '__main__':
    # Flag indicating if data has already been processed and saved
    existing_data = True
    # List of regions of interest for analysis
    regions_of_interest_dictionary = {'Right Frontal Lobe': [57, 59, 75, 77, 87, 95, 97, 121, 135],
                                      'Left Frontal Lobe': [58, 60, 76, 78, 88, 96, 98, 122, 136],
                                      'Right Temporal Lobe': [67, 89, 117, 131, 133, 137],
                                      'Left Temporal Lobe': [68, 90, 118, 132, 134, 138],
                                      'Right Occipital Lobe': [65, 79, 91, 93, 127],
                                      'Left Occipital Lobe': [66, 80, 92, 94, 128],
                                      'Right Hippocampal Complex': [19],
                                      'Left Hippocampal Complex': [20],
                                      'Brainstem': [7]}

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('../pickles/md_gm_longitudinal.pkl', 'rb') as f:
            [before_md_patients, after_md_patients, md_controls] = pkl.load(f)
        f.close()

    else:
        # Read data from an excel file
        file_path = r"/Users/leondailani/Documents/Part III Project/scan_data.ods"
        xlsx = pd.read_excel(file_path, sheet_name=None)

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['singleshell_malpem_FA_MD']

        # Form dataframes containing controls and patients and isolate MD data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'MD')
        # Get patients with known before/after data
        md_patients = multiple_scans(md_patients)
        # Get the extreme dataframes of the MD patients
        before_md_patients, after_md_patients = extreme_dataframes(md_patients)


        # Function to isolate the desired grey matter regions of interest (ROIs)
        def isolate_ROIs(df):
            # Iterating over each ROI to include it in the new dataframe
            for region, region_indices in regions_of_interest_dictionary.items():
                df[region] = 0
                # Some ROIs are the collection of multiple malpem regions and their values must be found additively.
                for index in region_indices:
                    df[region] = df[region] + df[index]
            df = df[regions_of_interest_dictionary.keys()]
            return df


        # Isolate and rename regions for the before, after and control dataframes
        before_md_patients, after_md_patients, md_controls = isolate_ROIs(before_md_patients), isolate_ROIs(
            after_md_patients), isolate_ROIs(md_controls)

        # Save the processed data to a pickle file
        with open('../pickles/md_gm_longitudinal.pkl', 'wb') as f:
            pkl.dump(
                [before_md_patients, after_md_patients, md_controls], f
            )
        f.close()

    # Define the different types of data of interest
    types = ['before', 'after', 'controls']
    type_dfs = [before_md_patients, after_md_patients, md_controls]

    # Remove outliers from the dataframe
    for i, type_df in enumerate(type_dfs):
        type_dfs[i] = outlier_removal(type_df)

    # Create an empty list to store the plot dataframes
    plot_dfs = []

    # We need to flip the table such that we have rows corresponding to each subject and column entries detailing the
    # type of scan and the region of interest, this is necessary for plotting using the seaborn library
    # Loop through each type of data and its corresponding dataframe
    for type, df in zip(types, type_dfs):
        # Create an empty list to store the dataframes for the same patient type and different regions
        same_type_dfs = []
        # Loop through each region of interest in the dictionary
        for region in regions_of_interest_dictionary.keys():
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

    # Select the rows of the dataframe that correspond to the non-cortical regions
    plot_df_non_cortical = plot_df.loc[(plot_df['region'] == 'Right Hippocampal Complex') |
                                       (plot_df['region'] == 'Left Hippocampal Complex') | (
                                               plot_df['region'] == 'Brainstem')]

    # Drop the rows of the dataframe that correspond to the cortical regions
    plot_df_cortical = plot_df.drop(plot_df_non_cortical.index.array)

    # Plotting the boxplot for the cortical regions:
    plot_df = plot_df_non_cortical
    sns.set(style='ticks', palette='pastel')

    # Increase the size of the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the plot
    sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax)

    # Add a title
    ax.set_title(
        "Mean Diffusivity in Non-Cortical Grey Matter Regions ",
        fontsize=20, fontweight='bold')

    # Set the x-axis label
    ax.set_xlabel("Region of the Brain", fontsize=15, fontweight='bold')

    # Set the y-axis label
    ax.set_ylabel("Mean Diffusivity Value (x10^-3 mm^2 s^-1)", fontsize=15, fontweight='bold')

    # Rotate the x-axis labels by 45 degrees
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45)

    # Add padding to the x-axis labels
    ax.margins(x=0.1)

    # Set the font size for the tick labels
    ax.tick_params(labelsize=12)

    # Add a grid to the plot
    sns.despine(offset=10, trim=True)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('gm_non_cortical.png')

    # Plotting the boxplot for the non-cortical regions:
    plot_df = plot_df_cortical
    sns.set(style='ticks', palette='pastel')

    # Increase the size of the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the plot with adjusted spacing
    sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax, dodge=0.5)

    # Add a title
    ax.set_title(
        "Mean Diffusivity in Cortical Grey Matter Regions",
        fontsize=20, fontweight='bold')

    # Set the x-axis label
    ax.set_xlabel("Region of the Brain", fontsize=15, fontweight='bold')

    # Set the y-axis label
    ax.set_ylabel("Mean Diffusivity Value (x10^-3 mm^2 s^-1)", fontsize=15, fontweight='bold')

    # Rotate the x-axis labels by 45 degrees and adjust font size
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, fontsize=12)

    # Add padding to the x-axis labels
    ax.margins(x=0.1)

    # Set the font size for the tick labels
    ax.tick_params(labelsize=12)

    # Add a grid to the plot
    sns.despine(offset=10, trim=True)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('gm_cortical.png')

    # Now want to construct a table with the t-metrics_and_atlases analyses


    # Create the DataFrame
    stats_df = pd.DataFrame({'Controls': np.median(md_controls, axis=0),
                             'Acute Covid-19': np.median(before_md_patients, axis=0),
                             'Longitudinal Covid-19': np.median(after_md_patients, axis=0)},
                            index=regions_of_interest_dictionary.keys())

    # Perform statistical tests and add p-values to the DataFrame
    comparisons_number = len(regions_of_interest_dictionary)
    stats_df['p-value for acute to longitudinal'] = stats.ttest_rel(after_md_patients, before_md_patients)[1]
    stats_df['p-value for longitudinal to controls'] = stats.ttest_ind(after_md_patients, md_controls)[1]

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Remove spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=False, left=False)

    # Plot the DataFrame as a table
    table = ax.table(cellText=stats_df.values.round(5), colLabels=stats_df.columns, rowLabels=stats_df.index,
                     loc='center', cellLoc='center')
    table.auto_set_column_width(col=list(range(len(stats_df.columns))))  # set cell width based on content
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # set font size
    table.scale(1, 1.5)  # scale the table
    plt.axis('off')  # hide axis
    plt.tight_layout()  # adjust layout to fit the table
    plt.savefig('md_gm_table.png')



