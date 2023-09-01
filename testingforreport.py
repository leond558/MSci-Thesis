import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from sheet_functions import boxplotter, most_significant_regions

with open('pickles/significant_and_corrections/FA_tract_table.pkl', 'rb') as f:
    [data,acute_md_patients,long_md_patients,md_controls] = pkl.load(f)
f.close()

# rois = list(data.keys())[:13]
# print(rois)

# acute_md_patients = long_md_patients[rois]
# md_controls = md_controls[rois]
#
# new = md_controls - acute_md_patients
# perc = (new/md_controls) * 100
# avg = perc.mean().mean()
# sem = stats.sem(perc.values.flatten(), nan_policy='omit')
#
# print(avg)
# print(sem)

region_num = 27

mean_a = np.mean(acute_md_patients)[region_num]
error_a = np.std(acute_md_patients)[region_num] / np.sqrt(len(acute_md_patients))

mean_l = np.mean(long_md_patients)[region_num]
error_l = np.std(long_md_patients)[region_num] / np.sqrt(len(long_md_patients))

# mean_l = np.mean(md_controls)[region_num]
# error_l = np.std(md_controls)[region_num] / np.sqrt(len(md_controls))

print(f'Acute: {mean_a-mean_l} Error: {error_a+error_l}')
# print(f'Long: {mean_l} Error: {error_l}')