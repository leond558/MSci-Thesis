import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns
from sheet_functions import multiple_scans, extreme_dataframes

if __name__ == '__main__':
    ''' Reading of the excel data each time is a computationally expensive process. Better to serialise and store the data after
    one read in a more efficient format such as using a pickle file.

    Have a marker indicating whether the data has already been transferred to pickle. If this marker isn't in place, proceeds with
    normal excel file reading.
    '''
    existing_data = True
    # File path containing the data
    file_path = r"/Users/leondailani/Documents/PART III PROJECT/data.ods"
    # Current sheet in the data to be investigated
    current_sheet_name = 'volumes'

    # If the data has already been transferred to a pickle file, read from that file.
    if existing_data:
        with open('pickles/' + current_sheet_name + '.pkl', 'rb') as f:
            [ss_controls_df, ss_patients_df, ms_controls_df,
             ms_patients_df] = pkl.load(f)
        f.close()

    # Otherwise, read data from the Excel sheet directly and write that to the
    else:
        xlsx = pd.read_excel(file_path, sheet_name=None)
        df = xlsx[current_sheet_name]


        # Multi-shell and single-shell data must be considered separately
        # Also necessary to separate control and patient data
        ms_controls_df = df.loc[(df['scheme'] == 'multi_shell') & (df['group'] == 'controls')].reset_index()
        ms_patients_df = df.loc[(df['scheme'] == 'multi_shell') & (df['group'] == 'patients')].reset_index()
        ss_controls_df = df.loc[(df['scheme'] == 'single_shell') & (df['group'] == 'controls')].reset_index()
        ss_patients_df = df.loc[(df['scheme'] == 'single_shell') & (df['group'] == 'patients')].reset_index()

        # Write relevant data to a pickle file
        with open('pickles/' + current_sheet_name + '.pkl', 'wb') as f:
            pkl.dump(
                [ss_controls_df, ss_patients_df, ms_controls_df, ms_patients_df],
                f)
        f.close()

    '''
    Here, we are interested in considering the patients that have before and after data. Want to consider the longitudinal
    impacts of covid-19 post hospitalisation. Hence, we construct a function that returns a data frame with
    subjects with multiple entries i.e. have multiple scans along a time period.
    '''

    '''
    Some patients have multiple scans not just a before and after. In this initial case, will only consider the most extreme time scans
    i.e. the earliest and the latest scans.
    '''

    # Considering the single shell calse
    ss_patients_df = multiple_scans(ss_patients_df)
    ss_patients_before_df, ss_patients_after_df = extreme_dataframes(ss_patients_df)

    # Isolate TotalBrain data only
    tot_ss_patients_before_df, tot_ss_patients_after_df = ss_patients_before_df['TotalBrain'], ss_patients_after_df[
        'TotalBrain']
    rows = [tot_ss_patients_after_df, tot_ss_patients_before_df]
    tot_ss_controls_df = ss_controls_df['TotalBrain']

    a, b = st.shapiro(tot_ss_patients_before_df)
    if b < 0.05:
        print("Not normally distributed. Shouldn't proceed with t-metrics_and_atlases")

    # Plotting boxplot
    sns.set()

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.boxplot(rows, vert=0)

    # Add a grid to the plot
    axs.grid(linestyle='--', alpha=0.7)

    # Set the color palette for the box plots
    colors = sns.color_palette()
    for i, box in enumerate(axs.artists):
        box.set_facecolor(colors[i])

    # Add labels and titles
    axs.set_title('Total Brain Volume in Earliest and Latest Scan', fontsize=18, fontweight='bold')
    plt.yticks([1, 2], ['After', 'Before'], fontsize=14)
    plt.xlabel('Volume/mm^3', fontsize=14)

    # Remove top and right spines
    sns.despine(top=True, right=True)

    # Adjust the plot layout
    plt.tight_layout()

    # Show the plot
    # plt.show()

    '''Now let's consider the region of the brain with the greatest change in volume and 
    the most statistically significant change in volume'''
    t_statistic, p_value = st.ttest_rel(ss_patients_after_df.loc[:, 'TotalBrain':],
                                        ss_patients_before_df.loc[:, 'TotalBrain':])
    most_significant = np.argmin(p_value)
    regions = list((ss_patients_after_df.loc[:, 'TotalBrain':]).columns)
    # print(p_value[most_significant])
    # print(regions[most_significant])
    # print(ss_patients_after_df['RightPP'].mean()-ss_patients_before_df['RightPP'].mean())

    percentage_significant = (len(p_value[p_value < 0.05]) / len(p_value)) * 100
    print(percentage_significant)
