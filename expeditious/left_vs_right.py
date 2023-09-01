import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from sheet_functions import boxplotter

fa_malpem = {'Left': 6, 'Right': 3}
md_tract = {'Left': 3, 'Right': 10}

import matplotlib.pyplot as plt
import pickle as pkl

d = md_tract

"""Looking at whether stratification is indeed present or merely a statistical anomaly."""

with open('../pickles/significant_and_corrections/MD_tract_table.pkl', 'rb') as f:
    [data,acute_md_patients,long_md_patients,md_controls] = pkl.load(f)
f.close()

lateral_rois = ['fronto-pontine tract left','fronto-pontine tract right', 'corticospinal tract left', 'corticospinal tract right' ]

acute_lateral = acute_md_patients[lateral_rois]
long_lateral = long_md_patients[lateral_rois]

difference_lateral = long_lateral -acute_lateral

stat_pontine, p_pontine_diff = wilcoxon(difference_lateral[lateral_rois[0]],difference_lateral[lateral_rois[1]])
stat_spinal, p_spinal_diff = wilcoxon(difference_lateral[lateral_rois[2]],difference_lateral[lateral_rois[3]])

stat_pontine_long, p_pontine_long = wilcoxon(long_lateral[lateral_rois[0]],long_lateral[lateral_rois[1]])
stat_spinal_long, p_spinal_long = wilcoxon(long_lateral[lateral_rois[2]],long_lateral[lateral_rois[3]])

mean_a = np.mean(acute_md_patients)
error_a = np.std(acute_md_patients) / np.sqrt(len(acute_md_patients))

mean_l = np.mean(long_md_patients)
error_l = np.std(long_md_patients) / np.sqrt(len(long_md_patients))


"""Make boxplot looking at the laterally paired regions looking at acute vs longitudinal data indvidiually."""

dfs = [acute_md_patients,long_md_patients]
types = ['Acute','Longitudinal']
rois = lateral_rois

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

sns.set(style='ticks', palette='pastel')

# Increase the size of the figure
fig, ax = plt.subplots(figsize=(10, 10))

# Create the plot
sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax)

# Add a title
ax.set_title('Investigating potential laterally stratified behaviour in white matter response to Covid-19 infection', fontweight='bold')

# Set the y-axis label
ax.set_ylabel('Mean Diffusivity $\mathregular{mm^{2}/s}$', fontweight='bold')

# Set the x-axis label
ax.set_xlabel('White Matter Tractseg ROI', fontweight='bold')

labels = [tick.get_text().title() for tick in ax.get_xticklabels()]
ax.set_xticklabels(labels)

# Customize grid
ax.grid(True, linestyle='--', linewidth=0.5)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Customize legend
ax.legend(title='Type', fontsize=10)

textstr = 'p-value of Wilcoxon test \n between longitudinal values \n for the Fronto-Pontine Fibres =  ' + str(p_pontine_long.round(3))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.15, 0.9, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

textstr = 'p-value of Wilcoxon test \n between longitudinal values \n for the Corticospinal Tract =  ' + str(p_spinal_long.round(3))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.65, .9, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('../plots/left_vs_right/paired_acute_long.png')
# plt.show()


"""Plotting the change in the acute vs lateral for each region."""
left_difference = difference_lateral.iloc[:,[0,2]]
right_difference = difference_lateral.iloc[:,[1,3]]

# Need to drop the left/right from the column names and the rois to make this work

left_difference_dict = {name.rsplit(' ', 1)[0]: left_difference[name] for name in left_difference.columns}
right_difference_dict = {name.rsplit(' ', 1)[0]: right_difference[name] for name in right_difference.columns}

left_difference = pd.DataFrame(left_difference_dict)
right_difference = pd.DataFrame(right_difference_dict)

diff = right_difference['fronto-pontine tract'] - left_difference['fronto-pontine tract']

mean_a = np.mean(right_difference)
error_a = np.std(diff) / np.sqrt(len(diff))
print(mean_a)


dfs = [left_difference,right_difference]
types = ['Left','Right']
rois = [" ".join(roi.split()[:-1]) for roi in lateral_rois]

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
fig, ax = plt.subplots(figsize=(10, 10))

# Create the plot
sns.boxplot(x='region', y='value', data=plot_df, hue='type', ax=ax)

# Add a title
ax.set_title('Investigating potential laterally stratified behaviour in white matter response to Covid-19 infection', fontsize=10, fontweight='bold')

# Set the y-axis label
ax.set_ylabel('Mean Diffusivity change between acute and longitudinal scans $\mathregular{{x10^{-5}}mm^{2}/s}$', fontsize=15, fontweight='bold')

# Set the x-axis label
ax.set_xlabel('White Matter Tractseg ROI', fontsize=15, fontweight='bold')

labels = [tick.get_text().title() for tick in ax.get_xticklabels()]
ax.set_xticklabels(labels)

# Customize grid
ax.grid(True, linestyle='--', linewidth=0.5)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Customize legend
ax.legend(title='Type', title_fontsize=12, fontsize=10)

textstr = 'p-value of Wilcoxon test \n of mean diffusivity change \n between left and right \n Fronto-Pontine Fibres =  ' + str(p_pontine_diff.round(3))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

textstr = 'p-value of Wilcoxon test \n of mean diffusivity change \n between left and right \n Corticospinal Tracts =  ' + str(p_spinal_diff.round(3))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.7, .9, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('../plots/left_vs_right/paired_difference.png')
# plt.show()

"""Formation of the plot showing the potential lateral stratification that exists in the response of the brain."""

# # Get the keys and values from the dictionary
# labels = d.keys()
# values = list(d.values())
#
# # Set the color scheme
# colors = ['#9C72B0', '#25A868']
#
# # Create a bar chart using Matplotlib
# fig, ax = plt.subplots(figsize=(10,10))
# rects = ax.bar(labels, values, color=colors)
#
# # Add a grid and set axis labels
# ax.grid(True, axis='y', linestyle='--', alpha=0.7)
# ax.set_ylabel('Number of statistically significant ROIs', fontsize=12)
# ax.set_xlabel('Side of the brain', fontsize=12)
# ax.set_title('Investigating which side of the brain has the most statistically significant ROIS for \n MD of white matter', fontsize=14)
#
# # Change the font size of the x-axis labels
# for label in ax.get_xticklabels():
#     label.set_fontsize(11)
#
# # Add the values as labels on top of the bars
# for i, rect in enumerate(rects):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2., height + 0.2, str(values[i]),
#             ha='center', va='bottom', fontsize=12, fontweight='bold')
#
# # Remove the top and right borders
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# textstr = 'p-value of binomial test: 0.0349'
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
#
# # Show the plot
# plt.tight_layout()
# # plt.savefig('../plots/left_vs_right/left_vs_right.png')
# # plt.show()

