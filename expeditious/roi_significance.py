import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem

from sheet_functions import boxplotter, most_significant_regions

with open('../pickles/significant_and_corrections/MD_tract_table.pkl', 'rb') as f:
    [data,acute_md_patients,long_md_patients,md_controls] = pkl.load(f)
f.close()

roi_number = 13
comparison_numbers = 72
significance_level = 0.05

def hochberg_adjustment(position):
    return significance_level/(comparison_numbers-(position+1)+1)
significance_levels = [hochberg_adjustment(i) for i in range(len(data))]


# reverse the p-values
reversed_data = {k: data[k] for k in reversed(data)}
significance_levels = np.array(significance_levels)
values = np.array(list(reversed_data.values()))

passes_threshold = values < significance_levels

first_significant = [i for i, x in enumerate(passes_threshold) if x][0]
new_significance_level = significance_levels[first_significant]
significant_rois = {k:v for k,v in zip(data.keys(),data.values()) if v < new_significance_level}


"""Plot for significance threshold and adjustment"""

# datavalues = [significance_levels,values]
# names = ['Hochberg Adjusted Significance Level','p-values']
# dict_of_values = {}
# for d,name in zip(datavalues,names):
#     dict_of_values[name] = d[30:]
#
# fig,ax = plt.subplots(figsize=(10,10))
#
# for k,v in dict_of_values.items():
#     ax.plot(v, label=k)
#
# ax.set_title('Determining statistical significance accounting for multiple comparisons using the Hochberg adjustment ')
# ax.set_xlabel('Reversed ordinality of the ROI in the list of most signifcantly changed regions')
# ax.set_ylabel('Value')
# ax.axvline(x=29, color='red', linestyle='--', label='Point at which null hypothesis can be rejected')
# ax.legend()
# x_labels = np.arange(30,72,8)
# ax.set_xticklabels(x_labels)
# plt.tight_layout()
# # plt.savefig('../plots/significant_and_corrections/Hochberg_Adjustment')
# # plt.show()

data = significant_rois
values = list(significant_rois.values())
rois = list(data.keys())[:roi_number]

'''Considering the change from the controls using a boxplot'''

change_controls = np.median(md_controls,axis=0)
change_acute = np.median(acute_md_patients,axis=0)
change_long = np.median(long_md_patients,axis=0)

# create a function to remove any nans in the array
def remove_nans(arr):
    mask = np.isnan(arr)
    return arr[~mask]

diff_a = remove_nans(change_acute-change_controls)
diff_l = remove_nans(change_long-change_controls)

fig, ax = plt.subplots(figsize=(10,10))
ax.boxplot([diff_a,diff_l])

# Assuming patient_index is the index of the patient in the diff_a and diff_l arrays
patient_index = 1  # replace with the actual patient index
ax.plot(1, diff_a[patient_index], 'go')
ax.plot(2, diff_l[patient_index], 'go')

ax.set_title('Investigating the change from the controls depending on whether data is acute or longitudinal from \n Covid-19 hospitalisation, for the regions of most significant acute change from the controls')
ax.set_ylabel('Change in mean diffusivity $\mathregular{{x}10^{-5}mm^{2}/s}$ relative to the controls ')
ax.set_xlabel('White matter Tractseg ROI')
ax.set_xticklabels(['Acute','Longitudinal'])

# add dashed line at y=0
ax.axhline(y=0, linestyle='--', color='gray')

plt.tight_layout()
plt.savefig('boop.png')
exit()
# # plt.savefig('../plots/change_from_controls/boxplot_MD_tract_boxplot_change.png')
# # plt.show()


"""Considering the change from the controls using a scatter plot"""

# # Regions that changed the most between controls and acute data
# d = most_significant_regions(md_controls,acute_md_patients,False,10)
# change_rois = list(d.keys())
#
# change_controls = np.median(md_controls[change_rois],axis=0)
# change_acute = np.median(acute_md_patients[change_rois],axis=0)
# change_long = np.median(long_md_patients[change_rois],axis=0)
#
# change_controls_error = np.std(md_controls[change_rois],axis=0)/np.sqrt(len(md_controls[change_rois]))
# change_acute_error = np.std(acute_md_patients[change_rois],axis=0)/np.sqrt(len(acute_md_patients[change_rois]))
# change_long_error = np.std(long_md_patients[change_rois],axis=0)/np.sqrt(len(long_md_patients[change_rois]))
#
# diff_a = change_controls-change_acute
# diff_l = change_controls-change_long
#
# error_diff_a = list(change_controls_error+change_acute_error)
# error_diff_l = list(change_controls_error+change_long_error)
#
#
#
# def adjusterror(err):
#     return err/10
#
# error_diff_a = [adjusterror(x) for x in error_diff_a]
# error_diff_l = [adjusterror(x) for x in error_diff_l]
#
# dict_a = {}
# dict_l = {}
#
# for name, a, l in zip(change_rois,diff_a,diff_l):
#     dict_a[name] = a
#     dict_l[name] = l
#
# # create scatter plot
# fig, ax = plt.subplots(figsize=(10,10))
# ax.errorbar(list(dict_a.keys()),list(dict_a.values()),yerr=error_diff_a, fmt='o', capsize=5, label='Acute',color='black')
# ax.errorbar(list(dict_l.keys()),list(dict_l.values()),yerr=error_diff_a, fmt='o', capsize=5, label='Longitudinal',color='blue')
#
# # plot errorbars
# # ax.errorbar(list(dict_a.keys()),)
#
# # add arrows between acute pairs and the 0 line
# for i, category in enumerate(dict_a.keys()):
#     x = list(dict_a.keys()).index(category)
#     y1 = dict_a[category]
#     offset = 0.15  # adjust offset as needed
#     ax.annotate("", xy=(x+offset, y1), xytext=(x+offset, 0), arrowprops=dict(arrowstyle="->",color='black'))
#
# # add arrows between long pairs and the 0 line
# for i, category in enumerate(dict_a.keys()):
#     x = list(dict_a.keys()).index(category)
#     y1 = dict_l[category]
#     offset = 0.25  # adjust offset as needed
#     ax.annotate("", xy=(x+offset, y1), xytext=(x+offset, 0), arrowprops=dict(arrowstyle="->",color='blue'))
#
# # Plotting difference arrow
# for i, category in enumerate(dict_a.keys()):
#     x = list(dict_a.keys()).index(category)
#     y1 = dict_a[category]
#     y2 = dict_l[category]
#     better = abs(y2) < abs(y1)
#     if better:
#         offset = 0.25  # adjust offset as needed
#         ax.annotate("", xy=(x+offset, y1), xytext=(x+offset, y2), arrowprops=dict(arrowstyle="-",color='green',label='Improved'))
#     else:
#         offset = 0.15  # adjust offset as needed
#         ax.annotate("", xy=(x + offset, y2), xytext=(x + offset, y1),
#                     arrowprops=dict(arrowstyle="-", color='red',label='Worsened'), label='worse')
#
# # add legend for arrow colors
# labels = ['Worsened','Improved']
# colors = ['red','green']
# dummy_handles = []
# for label , color in zip(labels,colors):
#     dummy_handles.append(ax.plot([], [],label=label, color=color)[0])
# ax.legend(handles=dummy_handles, labels=labels)
#
# # Tractseg titles
# # ax.set_title('Investigating the change from the controls depending on whether data is acute or longitudinal from \n Covid-19 hospitalisation, for the regions of most significant acute change from the controls')
# # ax.set_ylabel('Change in mean diffusivity $\mathregular{{x}10^{-3}mm^{2}/s}$ relative to the controls ')
# # ax.set_xlabel('White matter Tractseg ROI')
#
# # Malpem titles
# ax.set_title('Investigating the change from the controls depending on whether data is acute or longitudinal from \n Covid-19 hospitalisation, for the regions of most significant acute change from the controls')
# ax.set_ylabel('Change in fractional anisotropy relative to the controls ')
# ax.set_xlabel('Grey matter MALPEM ROI')
#
#
# # add dashed line at y=0
# ax.axhline(y=0, linestyle='--', color='gray')
#
#
# plt.xticks(rotation=45, ha='right')
#
# plt.tight_layout()
# plt.legend()
# # plt.savefig('../plots/change_from_controls/FA_malp_control_change.png')
# # plt.show()

""" Plot for table of significant regions and corresponding scan data values"""
# Create the DataFrame

# Isolate only the significant ROIS
md_controls = md_controls[rois]
acute_md_patients = acute_md_patients[rois]
long_md_patients = long_md_patients[rois]

stats_df = pd.DataFrame({'Controls': np.median(md_controls, axis=0),
                         'Acute Covid-19': np.median(acute_md_patients, axis=0),
                         'Longitudinal Covid-19': np.median(long_md_patients, axis=0)},
                        index=rois)


stats_df['p-value'] =values[:roi_number]
stats_df['adjusted significance level'] = new_significance_level

def format_number(num):
    return '{:.3g}'.format(num)

# Apply the custom formatting function to each cell in the DataFrame
stats_df = stats_df.applymap(format_number)

stats_df['significant?'] = 'Yes'

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(15, 15))

# Remove spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)

# Plot the DataFrame as a table
table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, rowLabels=stats_df.index,
                 loc='center', cellLoc='center')
table.auto_set_column_width(col=list(range(len(stats_df.columns))))  # set cell width based on content
table.auto_set_font_size(False)
table.set_fontsize(12)  # set font size
table.scale(1, 1.5)  # scale the table
plt.axis('off')  # hide axis
plt.tight_layout()  # adjust layout to fit the table
plt.savefig('malpem.png')
# plt.show()

""" Boxplot generator."""
# boxplotter([md_controls,acute_md_patients,long_md_patients],rois[:6],'Measuring the effect of Covid-19 hospitalisation on fractional anisotropy in grey matter regions', 'Grey Matter MALPEM ROI', 'Fractional Anisotropy','hello')
# $\mathregular{ms^{-2}}$
