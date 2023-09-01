# Importing necessary libraries
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Importing custom functions from the module 'sheet_functions'
from sheet_functions import multiple_scans, group_metric_split

if __name__ == '__main__':
    existing_data = True

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('../pickles/brainstem_evolution.pkl', 'rb') as f:
            [data,error] = pkl.load(f)
        f.close()

    else:
        # Read data from an excel file
        file_path = r"/Users/leondailani/Documents/Part III Project/scan_data.ods"
        xlsx = pd.read_excel(file_path, sheet_name=None)

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['singleshell_malpem_FA_MD']
        # Form dataframes containing controls and patients and isolate FA data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'FA')
        # Get patients with multiple scans that can be used to look into the evolution of the brainstem scan data
        md_patients = multiple_scans(md_patients,3)
        # isolate just the key information from those columns

        data = md_patients[['subject_id','scan_date',7]]
        data['scan_date'] = pd.to_datetime(md_patients['scan_date'], format='%Y%m%d')

        brainstem_data = md_patients[7]
        error = np.std(brainstem_data)/np.sqrt(len(brainstem_data))


        with open('../pickles/brainstem_evolution.pkl', 'wb') as f:
            pkl.dump(
                [data,error], f
            )
        f.close()

    # group by subject_id
    groups = data.groupby('subject_id')

    # make plot
    fig, ax = plt.subplots(figsize=(10,10))

    # give a plot colour
    colours = ['black','red','blue']

    # Loop over groups
    for (subject_id, group), color in zip(groups,colours):
        # Find the earliest and latest scan dates for this patient
        earliest_scan_date = group['scan_date'].min()
        group['scan_date'] = (group['scan_date'] -earliest_scan_date).dt.days

        # plot scatter points
        ax.scatter(group['scan_date'],group[7],label='Patient ' + str(subject_id), color=color)
        # plot error bars
        ax.errorbar(group['scan_date'],group[7], yerr=error, fmt=' ',color = color, capsize=5)

        # plot line of best fit through the scatter
        slope, intercept = np.polyfit(group['scan_date'],group[7],1)
        xfit = np.linspace(group['scan_date'].min()-10, group['scan_date'].max()+10, 200000)
        yfit = slope * xfit + intercept
        ax.plot(xfit,yfit, color=color)


    # set axes labes and title
    ax.set_ylabel('Fractional anisotropy value')
    ax.set_xlabel('Number of days since initial scan')

    ax2 = ax.twinx()

    # bar chart data
    scores = [38,1,11]
    times = [111,4,65]
    for score,time,color in zip(scores,times,colours):
        ax2.bar(time, score, color=color, alpha=0.5, width=10)

    ax2.set_ylabel('Verbal analogy cognitive metric score/ number correctly answered')
    ax2.tick_params(axis='y')


    plt.title('Investigating the change in FA value of the brainstem over time since initial scan \n whilst comparing against the verbal analogy cognitive metric')
    ax.legend()
    plt.savefig('../plots/brainstem_evolution/evolution_against_metric.png')
    plt.show()
