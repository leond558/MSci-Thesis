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
    existing_data = False

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('most_significant.pkl', 'rb') as f:
            [before_md_patients, after_md_patients, md_controls] = pkl.load(f)
        f.close()

    else:
        # Read data from an excel file
        file_path = r"/Users/leondailani/Documents/Part III Project/scan_data.ods"
        xlsx = pd.read_excel(file_path, sheet_name=None)

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['multishell_tractseg_FA_MD_MK_FW']

        # Form dataframes containing controls and patients and isolate MD data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'MD')
        # Get patients with known before/after data
        md_patients = multiple_scans(md_patients)
        # Get the extreme dataframes of the MD patients
        before_md_patients, after_md_patients = extreme_dataframes(md_patients)


        def getregions(df):
            regions = np.arange(72)
            df = df[regions]
            return(df)

        md_controls, before_md_patients, after_md_patients = getregions(md_controls), getregions(before_md_patients), getregions(after_md_patients)

        with open('most_significant.pkl', 'wb') as f:
            pkl.dump(
                [before_md_patients, after_md_patients, md_controls], f
            )
        f.close()

    print(before_md_patients)
    # p_values = stats.ttest_ind(md_controls,after_md_patients)[1]


    # p_values[np.isnan(p_values)] = 1
    # print(np.argsort(p_values))
    # print(np.argsort(stats.ranksums(md_controls,after_md_patients)[1]))


