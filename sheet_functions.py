# sheet function.py
import os
import numpy as np
from matplotlib import pyplot as plt
from segmentation_dicts.MALPEMDict import MALPEMDict
from segmentation_dicts.TractSegDict import TractSegDict
import pandas as pd
import seaborn as sns
from scipy import stats

"""This script comprises custom functions specifically designed for efficient data handling of dMRI and cognitive 
metric data. The functions serve the following purposes:

Division of data into acute and longitudinal timeframes. 
Identification and categorization of data into different scans per patient. 
Separation of data into fractional anisotropy or mean diffusivity. 
Removal of outliers from the data.
Calculation of statistical significance in the change between two dataframes.
Application of parcellation to the data and assignment of ROI names.
Plotting data in the form of boxplots.

These functions streamline the data handling process and facilitate effective analysis of dMRI and cognitive 
metric data."""


def multiple_scans(df, number: int = 2):
    """
    Function that takes a dataframe and outputs a new dataframe with entries that have subjects that have scanned
    multiple times.
    :param df: Dataframe that takes structural mri scan data with a format corresponding to the Excel data sheet
    :param number: Minimum number of multiple scans to be included
    :return: A new dataframe with entries corresponding to multiple scans by the same subject, for different subjects.
    """
    # Creating a boolean array with markers corresponding to the position of entries from subjects who have multiple
    # scans
    multiple = (df.groupby('subject_id')['subject_id'].transform('size') >= number)
    df_multiple = df[multiple]
    # Restructuring and removing inconsequential data
    df_multiple = df_multiple.drop(['index', 'scheme', 'group'], axis=1)
    df_multiple = df_multiple.sort_values(by=['subject_id']).reset_index()
    return df_multiple


def initial_scans(df):
    """
    Function that returns a dataframe with only the first ever scans completed by a patient.
    :param df: Dataframe contating patient scan data with the scan_date column.
    :return: New dataframe with just the first ever scans performed.
    """
    # Convert scan date column to datetime
    df['scan_date'] = pd.to_datetime(df['scan_date'], format='%Y%m%d')

    # Create two new dataframes to hold earliest and latest scans
    initials = pd.DataFrame()

    # Group scans by subject ID
    groups = df.groupby('subject_id')

    # Loop over groups
    for subject_id, group in groups:
        # Find the earliest and latest scan dates for this patient
        earliest_scan_date = group['scan_date'].min()
        initials = pd.concat([initials, group[group['scan_date'] == earliest_scan_date]])

    initials = initials.reset_index()
    initials = initials.drop(['index', 'level_0'], axis=1)

    return initials


def extreme_dataframes(df):
    """
    Function that creates a new dataframe that contains only the latest and earliest scans from each patient and not
    any intermediary scans.
    :param df: Takes a dataframe that contains only patient data with patients that have multiple scans.
    :return: A tuple of dataframes. One is a dataframe of patient scans containing their earliest scan and the other
    is a dataframe that contains their latest scan.
    """
    # Group by individual patient

    subject_grouped = df.groupby('subject_id')
    # An array of the index positions of the extreme time entries in the original data frame
    extreme_indices = []
    # Find these index positions
    for name, group in subject_grouped:
        before_index = group['scan_date'].astype(int).idxmin()
        after_index = group['scan_date'].astype(int).idxmax()
        extreme_indices.append((before_index, after_index))

        # Construct the new dataframes with the before and after scans
        before_rows = df.loc[[i[0] for i in extreme_indices]].reset_index()
        after_rows = df.loc[[i[1] for i in extreme_indices]].reset_index()

        # Remove artefacts introduced in the dataframe
        before_rows = before_rows.drop(['index', 'level_0'], axis=1)
        after_rows = after_rows.drop(['index', 'level_0'], axis=1)

    return before_rows, after_rows


def group_metric_split(df, metric):
    """
    Function that separates the existing dataframe into two dataframes with controls and patients
    respectively and isolates only entries corresponding to the specific provided metric.
    :param df: Dataframe containing the diffusion tensor imaging data in a form that aligns with
    the data spreadsheet.
    :param metric: The desired metric of diffusion tensor imaging to be considered
    :return: A tuple of a controls data frame and a patients dataframe along the provided metric.
    """
    possible_metrics = ['MD', 'FA', 'MK', 'FW']
    if metric not in possible_metrics:
        raise Exception("Invalid metric requested")
    df_controls = df.loc[(df['group'] == 'controls') & (df['metric'] == metric)].reset_index()
    df_patients = df.loc[(df['group'] == 'patients') & (df['metric'] == metric)].reset_index()
    return df_controls, df_patients


def limited_timed_dataframes(df: pd.DataFrame, limit: int):
    """
    A function that takes in a dataframe containing scan data for patients with multiple scans. The function
    returns two new dataframes containing the scan data corresponding to the earliest conducted scan and the
    latest conducted scan for each patient, provided that that latest scan happened less than the provided time
    limit away from the earliest one. If no such latest scan be identified that adheres to the time limit,
    that patient's data is excluded.
    :param df: Dataframe with scan data containing 'scan_date' and 'subject_id'
    data.
    :param limit: The time limit for which longitudinal scans must have happened after the initial earliest
    acute scan.
    :return: Two new dataframes. One with the acute scan data for patients and the other for the
    longitudinal scan data. Provided the longitduinal data is within the limit.
    """
    # Convert scan date column to datetime
    df['scan_date'] = pd.to_datetime(df['scan_date'], format='%Y%m%d')

    # Create two new dataframes to hold earliest and latest scans
    acute_scans = pd.DataFrame()
    long_scans = pd.DataFrame()

    # Group scans by subject ID
    groups = df.groupby('subject_id')

    # Loop over groups
    for subject_id, group in groups:
        # Find the earliest and latest scan dates for this patient
        earliest_scan_date = group['scan_date'].min()
        latest_scan_date = group['scan_date'].max()

        # Check if the latest scan happened more than a year after the earliest scan
        if (latest_scan_date - earliest_scan_date).days > limit:
            # Find the intermediate scans that happened within one year of the earliest scan
            intermediate_scans = group[
                (group['scan_date'] > earliest_scan_date) & (group['scan_date'] < latest_scan_date) & (
                        (group['scan_date'] - earliest_scan_date).dt.days <= limit)]

            # Find the intermediate scan that happened the furthest in time from the earliest scan
            # If there are no intermediate scans and the latest scan exceeds the time limit between scans,
            # do not add any entry to either acute_scans or long_scans dataframes
            if len(intermediate_scans) > 0:
                furthest_scan = intermediate_scans.sort_values(by='scan_date', ascending=False).iloc[0]

                # Add the furthest intermediate scan to the latest scans dataframe
                long_scans = pd.concat([long_scans, pd.DataFrame(furthest_scan).transpose()])

                # Add the earliest scan to the earliest scans dataframe
                acute_scans = pd.concat([acute_scans, group[group['scan_date'] == earliest_scan_date]])

        else:
            # Add the latest scan to the latest scans dataframe
            long_scans = pd.concat([long_scans, group[group['scan_date'] == latest_scan_date]])

            # Add the earliest scan to the earliest scans dataframe
            acute_scans = pd.concat([acute_scans, group[group['scan_date'] == earliest_scan_date]])

    # Remove artefacts introduced in the dataframe
    long_scans = long_scans.reset_index()
    acute_scans = acute_scans.reset_index()

    acute_scans = acute_scans.drop(['index', 'level_0'], axis=1)
    long_scans = long_scans.drop(['index', 'level_0'], axis=1)

    return acute_scans, long_scans


def outlier_removal(df):
    """
    Function that removes any data points beyond 3 standard deviations of the mean.
    This is done through calculation of a z-score.
    :param df: Dataframe containing data.
    :return: New dataframe with outliers removed
    """
    df = df[(np.abs(stats.zscore(df)) < 6).all(axis=1)]
    return df


def isolate_regions(df: pd.DataFrame, segmentation: str, keepid: bool = False, keepdatediff=False):
    """
    Function that returns a dataframe with columns containing only scan data. No additional patient identifier or
    other similar supplementary data is contained.
    :param keepdatediff: True/False keep the date_diff column in the dataframe
    :param keepid: True/False keep the subject_id in the dataframe
    :param df: Dataframe containing scan data and other data.
    :param segmentation: The segmentation framework that the data is in. Can be either 'malpem', 'tractseg' or 'jhu'.
    :return: A dataframe with only scan data and the column names corresponding to brain region.
    """
    # Change input to lowercase for recgonisability
    segmentation = segmentation.lower()
    # Depending on which segmentation framework is specified, create an object containing the dictionary of that
    # framework. The dictionary contains a key: value relationship relating index to brain region.

    # If we want to preserve the subject_id column, keep a temporary copy of the original dataframe
    if keepid:
        temp_df = df

    # Determine which segmentation framework is to be used
    if segmentation == 'malpem':
        region_dict = MALPEMDict.malpem_dict
    elif segmentation == 'tractseg':
        region_dict = TractSegDict.tractseg_dict

    # If an invalid segmentation framework is passed as an input, raise an exception.
    else:
        raise Exception('Invalid brain segmentation protocol given.')

    # Isolate only columns in the provided dataframe that correspond with scan data brain regions
    df = df[list(region_dict.keys())]
    # Rename the columns with the names of the brain regions.
    df = df.rename(columns=region_dict)
    df.columns = map(str.lower, df.columns)

    # Reinsert subject_id column through copying it from the temporary dataframe copy, if necessary
    if keepid:
        df['subject_id'] = temp_df['subject_id']

    if keepdatediff:
        df['date_diff'] = temp_df['date_diff']

    return df


def boxplotter(dfs: list, rois: list, title: str = '',
               x_label: str = '', y_label: str = '', save_name: str = '',
               types: list = ['controls', 'acute', 'longitudinal']):
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

    print(plot_df)

    sns.set(style='ticks', palette='pastel')

    # Increase the size of the figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Create the plot
    sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax)
    max_y = plot_df['value'].max()
    min_y = plot_df['value'].min()

    ax.set_ylim([min_y * 0.85, max_y * 1.1])

    # Add a title
    ax.set_title(title, fontsize=20, fontweight='bold')

    # Set the x-axis label
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')

    # Set the y-axis label
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold')

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

    filename = os.path.basename(__file__)[:-3]
    plt.savefig('../plots/' + save_name + '.png')

    plt.show()


def most_significant_regions(df1: pd.DataFrame, df2: pd.DataFrame, related: bool, number_rois: int):
    """
    Determines the p-values for rois conducting the necessary test given if the data is related or not.
    :param df1: Dataframe 1
    :param df2: Dataframe 2
    :param related: Boolean to inform whether the dataframes are related or not.
    :param number_rois: Number of rois that are needed.
    :return: Returns a dictionary with keys of roi names and p-values corresponding to the change in the value between
    the two dataframes
    """
    # If the two samples are related, need to perform a Signed-Rank metrics_and_atlases.
    if related:
        # Doing a basic check to see if the groups are indeed related. If related, they should have the same number of
        # entries.
        if len(df1) != len(df2):
            # Raise an exception if this is not the case
            raise Exception('These two dataframes are not of related groups as they contain different number of '
                            'entries.'
                            'The group size is not the same.')
        # Perform the Signed-Rank metrics_and_atlases. p is the p-value.
        stat, p = stats.wilcoxon(df1, df2)

    else:
        # If two samples are independent, want to perform a Wilcoxon Rank Sum Test instead
        stat, p = stats.ranksums(df1, df2)

    # Replace nan values with 1
    p = np.nan_to_num(p, nan=1)

    # Want to isolate and return the top most statistically significant regions for changes
    # First sort the regions in ascending order (smallest p-value first) and get the indexes of those
    # 15 lowest p-value regions
    most_significant_ps = np.sort(p)[:number_rois]
    most_significant_indices = np.argsort(p)[:number_rois]

    # Now want to recover what each region these indices correspond to are.
    # First need to figure out which brain segmentation framework is being used based on length of dictionaries.
    # / the number of columns in the dataframes.

    # Need to add one because malpem segmentation framework dictionaries start from a key value of 1
    # but argsort starts from an index of 0. Tractseg starts from 0, so can avoid doing this.

    number_columns = df1.shape[1]
    if number_columns == len(MALPEMDict.malpem_dict):
        get_dict_value = np.vectorize(lambda key: MALPEMDict.malpem_dict[key])
        most_significant_indices = most_significant_indices + 1

    elif number_columns == len(TractSegDict.tractseg_dict):
        get_dict_value = np.vectorize(lambda key: TractSegDict.tractseg_dict[key])

    most_significant_rois = get_dict_value(most_significant_indices)

    # Create a dictionary linking most significantly changed ROIs as keys with the corresponding value being
    # the p-value
    roi_p_dict = dict(zip(np.char.lower(most_significant_rois), most_significant_ps))

    return roi_p_dict
