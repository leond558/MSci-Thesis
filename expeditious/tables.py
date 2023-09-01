import pickle as pkl
import numpy as np
import pandas as pd
from segmentation_dicts.TractSegDict import TractSegDict
from segmentation_dicts.MALPEMDict import MALPEMDict
from math import log10, floor
import re

"""
Function to create a table that recreates that found in the supplementary materials section of the acute study.
This function is for the WHITE MATTER regions using the tractseg atlas.
"""


# choose either md or fa for white matter
def white_matter_table_maker(metric):
    metric = metric.lower()

    # magnitude integer determines whether the values are scaled by an order of magnitude for readability purposes
    if metric == 'md':
        metric_string = 'MD_tract'
        magnitude = 1000
    elif metric == 'fa':
        metric_string = 'FA_tract'
        magnitude = 1
    else:
        raise Exception("Metric must either be 'MD' or 'FA. Invalid input.")

    with open('../pickles/significant_and_corrections/' + metric_string + '_table.pkl', 'rb') as f:
        [data, acute_md_patients, long_md_patients, md_controls] = pkl.load(f)
    f.close()

    table_df = pd.DataFrame()

    region_dict = TractSegDict.tractseg_dict
    rois = list((region_dict.values()))
    rois = [roi.title() for roi in rois]

    table_df['ROI'] = rois

    # define a function to round numbers
    def significant_rounder(number, figures=2):
        return round(number, figures - int(floor(log10(abs(number)))) - 1)

    # identifying the range of values

    def median_and_range_stringer(df):
        max_vals = [x * magnitude for x in list(df.max())]
        min_vals = [x * magnitude for x in list(df.min())]
        med_vals = [x * magnitude for x in list(df.median())]

        table_entries = []
        for max, min, med in zip(max_vals, min_vals, med_vals):
            table_entry = str(significant_rounder(med, 2)) + " (" + str(significant_rounder(min, 2)) + "-" + str(
                significant_rounder(max, 2)) + ")"
            table_entries.append(table_entry)
        return table_entries

    # Add the scan data

    table_df['Controls'] = median_and_range_stringer(md_controls)
    table_df['Acute'] = median_and_range_stringer(acute_md_patients)
    table_df['Longitudinal'] = median_and_range_stringer(long_md_patients)

    # Adding the p-values

    roi_order = [region.lower() for region in list(region_dict.values())]
    ordered_data = dict(sorted(data.items(), key=lambda x: roi_order.index(x[0])))

    table_df['p-value'] = [significant_rounder(x, 2) for x in list(ordered_data.values())]

    # Finding the adjusted significance levels

    comparison_numbers = 72
    significance_level = 0.05

    # Finding the adjusted significance levels for regions that are rejected according to the Hochberg adjustment

    def hochberg_adjustment(position):
        return significance_level / (comparison_numbers - (position + 1) + 1)

    significance_levels = np.array([hochberg_adjustment(i) for i in range(len(data))])
    adjusted_significance_levels_dict = {k: v for k, v in zip(data.keys(), significance_levels)}
    ordered_significance_levels = dict(
        sorted(adjusted_significance_levels_dict.items(), key=lambda x: roi_order.index(x[0])))

    # Changing the adjusted significance threshold to that when comparison stops and the conclusion of null
    # hypothesis rejection can be applied

    reversed_data = {k: data[k] for k in reversed(data)}
    values = np.array(list(reversed_data.values()))

    passes_threshold = values < significance_levels

    first_significant = [i for i, x in enumerate(passes_threshold) if x][0]
    new_significance_level = significance_levels[first_significant]
    significant_rois = [k for k, v in zip(data.keys(), data.values()) if v < new_significance_level]

    # replacing the significance threshold with the alternative value from the adjustment
    for region in significant_rois:
        if region in ordered_significance_levels:
            ordered_significance_levels[region] = new_significance_level

    table_df['Adjusted Significance Threshold'] = [significant_rounder(x, 2) for x in
                                                   list(ordered_significance_levels.values())]

    # Establish which regions are significant and output them to a text file

    with open('/outputs' + metric_string + '.txt', mode='w') as file:
        file.write(str(significant_rois))

    # Make groupings

    new_df = pd.concat(
        [group.reset_index(drop=True) for _, group in table_df.groupby(table_df['ROI'].str.contains('Thala'))])

    # Output table to excel

    new_df.to_excel('/outputs' + metric_string + '.xlsx', index=False)

    return


"""
Function to create a table that recreates that found in the supplementary materials section of the acute study.
This function is for the GREY MATTER regions using the tractseg atlas.
"""


def grey_matter_table_maker(metric, save_table = False):
    metric = metric.lower()

    # magnitude integer determines whether the values are scaled by an order of magnitude for readability purposes
    if metric == 'md':
        metric_string = 'MD_malp'
        magnitude = 1000
    elif metric == 'fa':
        metric_string = 'FA_malp'
        magnitude = 1
    else:
        raise Exception("Metric must either be 'MD' or 'FA. Invalid input.")

    # Get the patient data

    with open('../pickles/significant_and_corrections/' + metric_string + '_table.pkl', 'rb') as f:
        [data, acute_md_patients, long_md_patients, md_controls] = pkl.load(f)
    f.close()

    # Get the propeitary MALPEM parcellation used in the acute study

    # Read data from an excel file
    file_path = r"/Users/leondailani/Documents/Part III Project/MALPEM_parcellation.xlsx"
    xlsx = pd.read_excel(file_path, sheet_name=None)

    # Get single-shell malpem DTI data from the loaded excel file
    malpem_groupings = xlsx['MALP-EM_parcellation']

    malpem_groupings = malpem_groupings[['ROI Index', 'Grouping']]

    # map ROI index in groupings to the names of the regions used in the dict
    region_dict = MALPEMDict.malpem_dict

    malpem_groupings['ROI Index'] = malpem_groupings['ROI Index'].map(region_dict)

    # make all the entries lowercase in order to be compatible with the data tables
    malpem_groupings = malpem_groupings.applymap(lambda x: x.lower() if type(x) == str else x)

    # Convert the table into a dictionary
    malpem_groupings = malpem_groupings.set_index('ROI Index')['Grouping'].to_dict()

    # Define a list of the regions used in the study and simplify the table to only include those regions
    study_regions = ['right frontal lobe', 'left frontal lobe', 'right temporal lobe', 'left temporal lobe',
                     'right parietal lobe', 'left parietal lobe', 'right occipital lobe', 'left occipital lobe',
                     'right hippocampal complex', 'left hippocampal complex', 'mesencephalic reticular formation',
                     'oral pons']

    # Define a sub-list that accounts for the regions that will go into the total grey matter calculation

    total_gm_regions = study_regions[:-4]

    # Create new patient and control data tables with the columns being the groupings and the entries being the
    # averages that went into forming those groupings. Create a function for this so that it can applied efficiently
    # to each of the dataframes.

    def impose_groupings(df):
        '''
        Function that changes the table to the propietary GM groupings used in the acute study.
        :param df: Dataframe containing patient data in the original MALPEM atlas columns
        :return: Dataframe with the new study groupings
        '''

        # Impose groupings
        df = df.groupby(malpem_groupings, axis=1).mean()

        # Isolate only relevant regions included in the acute study and create total column
        df = df.loc[:,study_regions]
        df['total supatentorial grey matter'] = df[total_gm_regions].mean(axis=1)

        # Move total column to the front
        total_col = df.pop('total supatentorial grey matter')
        df.insert(0,'total supatentorial grey matter',total_col)

        return df

    acute_md_patients = impose_groupings(acute_md_patients)
    long_md_patients = impose_groupings(long_md_patients)
    md_controls = impose_groupings(md_controls)

    mean_a = np.mean(acute_md_patients['right occipital lobe'])
    error_a = np.std(acute_md_patients['right occipital lobe'])/np.sqrt(len(acute_md_patients))

    mean_l = np.mean(long_md_patients['right occipital lobe'])
    error_l = np.std(long_md_patients['right occipital lobe']) / np.sqrt(len(acute_md_patients))

    mean_c = np.mean(md_controls['right occipital lobe'])
    error_c = np.std(md_controls['right occipital lobe']) / np.sqrt(len(md_controls))

    print(f'Acute change: {100*((mean_a-mean_c)/mean_c)}, Error:{np.sqrt((error_a+error_c)**2+error_c**2)}')
    print(f'Long change: {100*((mean_l-mean_c)/mean_c)}, Error:{np.sqrt((error_l+error_c)**2+error_c**2)}')

    exit()
    # If using this table for further analytics, want to write to a pickle

    if save_table:

        with open('../pickles/study_histogram.pkl', 'wb') as f:
            pkl.dump(
                [acute_md_patients,long_md_patients,md_controls], f
            )
        f.close()

    table_df = pd.DataFrame()

    rois = [roi.title() for roi in acute_md_patients.columns]

    table_df['ROI'] = rois

    # define a function to round numbers
    def significant_rounder(number, figures=2):
        return round(number, figures - int(floor(log10(abs(number)))) - 1)

    # identifying the range of values

    figures = 3
    def median_and_range_stringer(df):
        max_vals = [x * magnitude for x in list(df.max())]
        min_vals = [x * magnitude for x in list(df.min())]
        med_vals = [x * magnitude for x in list(df.median())]

        table_entries = []
        for max, min, med in zip(max_vals, min_vals, med_vals):
            table_entry = str(significant_rounder(med, figures)) + " (" + str(significant_rounder(min, figures)) + "-" + str(
                significant_rounder(max, figures)) + ")"
            table_entries.append(table_entry)
        return table_entries

    # Add the scan data

    table_df['Controls'] = median_and_range_stringer(md_controls)
    table_df['Acute'] = median_and_range_stringer(acute_md_patients)
    table_df['Longitudinal'] = median_and_range_stringer(long_md_patients)


    # Adding the p-values

    roi_order = [region.lower() for region in list(region_dict.values())]
    ordered_data = dict(sorted(data.items(), key=lambda x: roi_order.index(x[0])))

    # Accounting for the propietary groupings used in the acute study

    ordered_data_df = pd.DataFrame(ordered_data,index=[0])
    ordered_data_df = impose_groupings(ordered_data_df)

    # Converting back into the desired dictionary format

    ordered_data = ordered_data_df.to_dict(orient='records')[0]

    # Inserting the p-value into the table

    table_df['p-value'] = [significant_rounder(x, figures) for x in list(ordered_data.values())]

    # Redefining the data and roi_order variable with the study groupings

    data = table_df.set_index('ROI')['p-value'].to_dict()

    roi_order = [region for region in list(data.keys())]

    data = dict(sorted(data.items(), key=lambda x: x[1]))

    # Finding the adjusted significance levels

    comparison_numbers = 13
    significance_level = 0.05

    # Finding the adjusted significance levels for regions that are rejected according to the Hochberg adjustment

    def hochberg_adjustment(position):
        return significance_level / (comparison_numbers - (position + 1) + 1)

    significance_levels = np.array([hochberg_adjustment(i) for i in range(len(data))])
    adjusted_significance_levels_dict = {k: v for k, v in zip(data.keys(), significance_levels)}
    ordered_significance_levels = dict(
        sorted(adjusted_significance_levels_dict.items(), key=lambda x: roi_order.index(x[0])))

    # Changing the adjusted significance threshold to that when comparison stops and the conclusion of null
    # hypothesis rejection can be applied

    reversed_data = {k: data[k] for k in reversed(data)}
    values = np.array(list(reversed_data.values()))

    passes_threshold = values < significance_levels

    # If no regions surpass the threshold then don't do anything

    if any(passes_threshold):

        first_significant = [i for i, x in enumerate(passes_threshold) if x][0]
        new_significance_level = significance_levels[first_significant]
        significant_rois = [k for k, v in zip(data.keys(), data.values()) if v < new_significance_level]

        # replacing the significance threshold with the alternative value from the adjustment
        for region in significant_rois:
            if region in ordered_significance_levels:
                ordered_significance_levels[region] = new_significance_level

        # Establish which regions are significant and output them to a text file

        with open('outputs/' + metric_string + '.txt', mode='w') as file:
            file.write(str(significant_rois))


    table_df['Adjusted Significance Threshold'] = [significant_rounder(x, 2) for x in
                                                   list(ordered_significance_levels.values())]

    # Output table to excel

    table_df.to_excel('outputs/'+ metric_string + '.xlsx', index=False)

    return


grey_matter_table_maker('md',True)

# region_dict = MALPEMDict.malpem_dict

# def region_matcher(region_identifier,lateral = False, lateral_string = ''):
#     word = region_identifier
#     pattern = re.compile(rf"\b\w*{word}\w*\b", re.IGNORECASE) # regex pattern to match the target word
#     matching_regions = [value for key, value in region_dict.items() if pattern.search(value.lower())]
#     if lateral:
#         return [region for region in matching_regions if lateral_string in region.lower()]
#     else:
#         return matching_regions
#
# frontal_right = region_matcher('frontal','right')
# frontal_right = region_matcher('frontal','right')
# temporal_right = region_matcher('temporal','right')
# temporal_right = region_matcher('temporal','right')
