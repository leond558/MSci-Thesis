# Importing necessary libraries
import os

import numpy as np
import pandas
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# Importing custom functions from the module 'sheet_functions'
from sheet_functions import multiple_scans, group_metric_split, limited_timed_dataframes, isolate_regions, boxplotter, most_significant_regions

if __name__ == '__main__':
    existing_data = False

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('pickles/prelim_multishell.pkl', 'rb') as f:
            [md_controls, acute_md_patients, long_md_patients] = pkl.load(f)
        f.close()

    else:
        # Read data from an excel file
        file_path = r"/Users/leondailani/Documents/Part III Project/scan_data.ods"
        xlsx = pd.read_excel(file_path, sheet_name=None)

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['multishell_malpem_FA_MD_MK_FW']
        # Form dataframes containing controls and patients and isolate MD data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'MD')
        # Get patients with known before/after data
        md_patients = multiple_scans(md_patients)
        # # Get the extreme dataframes of the MD patients
        acute_md_patients, long_md_patients = limited_timed_dataframes(md_patients, 500)
        md_controls, acute_md_patients, long_md_patients = isolate_regions(md_controls, 'malpem'), isolate_regions(
            acute_md_patients, 'malpem'), isolate_regions(long_md_patients, 'malpem')
        with open('pickles/prelim_multishell.pkl', 'wb') as f:
            pkl.dump(
                [md_controls, acute_md_patients, long_md_patients], f
            )
        f.close()

    # type_dfs = [acute_md_patients, long_md_patients, md_controls]
    # boxplotter( type_dfs, ['BrainStem',"RightAmygdala"], 'Mean Diffusivity in the Brainstem', 'Region of the brain',
    #            'Mean diffusivity', 'brainstemtest')

    d = most_significant_regions(acute_md_patients,long_md_patients,True,138)
    print(d)

    with open('pickles/significant_and_corrections/MD_malp_table.pkl', 'wb') as f:
        pkl.dump(
            [d,acute_md_patients,long_md_patients,md_controls], f
        )
    f.close()

    # print(most_significant_regions(long_md_patients,md_controls,False,10))
    # print(most_significant_regions(acute_md_patients,md_controls,False,10))