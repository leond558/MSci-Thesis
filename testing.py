import pickle as pkl
import pandas as pd
import numpy as np
from scipy.stats import norm, kstest, shapiro, anderson
import matplotlib.pyplot as plt

from sheet_functions import boxplotter, multiple_scans, group_metric_split, isolate_regions

if __name__ == '__main__':
    existing_data = True

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('pickles/testing.pkl', 'rb') as f:
            [data] = pkl.load(f)
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

        data = md_patients[['subject_id','scan_date',18,14,19,31]]
        data['scan_date'] = pd.to_datetime(md_patients['scan_date'], format='%Y%m%d')
        data['time'] = data.groupby('subject_id')['scan_date'].transform(lambda x: (x - x.min()).dt.days)
        data = data.drop('scan_date',axis=1)

        with open('pickles/testing.pkl', 'wb') as f:
            pkl.dump(
                [data], f
            )
        f.close()

keys = [18,14,19,31]
values = ['fronto_pontine_tract_left', 'corticospinal_tract_left', 'fronto_pontine_tract_right', 'parieto_occipital_pontine_left']
my_dict = {k: v for k, v in zip(keys, values)}
data = data.rename(columns = my_dict)

data.to_csv(r'/Users/leondailani/Documents/Part III Project/longitudinal.csv',index=False)