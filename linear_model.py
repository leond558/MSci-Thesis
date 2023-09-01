# Importing necessary libraries

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import kstest, shapiro, norm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from dataframe_classes.CognitiveMetrics import CognitiveMetrics
from dataframe_classes.SystematicComponents import SystematicComponents
# Importing custom functions from the module 'sheet_functions'
from sheet_functions import group_metric_split, isolate_regions, multiple_scans, \
    limited_timed_dataframes

if __name__ == '__main__':
    # Marker for whether to use existing data or not
    existing_data = False
    # Marker for whether to include other systematic components in the dataframe like age etc
    consider_systematics = True
    #  Marker for whether to include only statistically significantly changing and literature motivated brain regions
    only_significant = True

    # Sets the currently considered cognitive metric
    current_metric = 'score'
    metric_in_sheet = "    {'rs_targetDetection_RT'                   }"

    if existing_data:
        # Load pre-processed data from a pickle file
        with open('pickles/linear_model.pkl', 'rb') as f:
            [multi_difference_merged, initial_merged] = pkl.load(f)
        f.close()

    else:
        # Read data from an excel file
        file_path = r"/Users/leondailani/Documents/Part III Project/scan_data.ods"
        xlsx = pd.read_excel(file_path, sheet_name=None)

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['multishell_malpem_FA_MD_MK_FW']

        # Form dataframes containing controls and patients and isolate MD data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'FA')

        # Get the difference dataframe
        # Get patients with known before/after data
        md_patients_multiple = multiple_scans(md_patients)
        # # Get the extreme dataframes of the MD patients
        acute_md_patients, long_md_patients = limited_timed_dataframes(md_patients_multiple, 500)

        # Creating a dataframe for the difference in brain scan results
        acute_md_patients_temp = acute_md_patients.drop(['project_id', 'scan_date', 'scan_id', 'metric'], axis=1)
        long_md_patients_temp = long_md_patients.drop(['project_id', 'scan_date', 'scan_id', 'metric'], axis=1)

        difference_md_patients = long_md_patients_temp.subtract(acute_md_patients_temp)

        # Inserting days between scans
        difference_md_patients['date_diff'] = (long_md_patients['scan_date'] - acute_md_patients['scan_date']).dt.days

        if not ((difference_md_patients['subject_id'] == 0).all()):
            raise Exception('Difference dataframe not calculated properly')

        difference_md_patients = difference_md_patients.drop('subject_id', axis=1)
        difference_md_patients['subject_id'] = long_md_patients['subject_id']


        # md_patients = md_patients.drop(['project_id', 'scan_id', 'metric', 'group', 'scheme'], axis=1)
        # initial_md_patients = initial_scans(md_patients)
        # initial_md_patients = initial_md_patients.drop(['scan_date'], axis=1)

        # Insert region names into columns and preserve the subject_id
        difference_md_patients = isolate_regions(difference_md_patients, 'malpem', True, True)
        # initial_md_patients = isolate_regions(initial_md_patients, 'malpem', True)


        # Only have statistically significant regions in the dataframe
        # other closely significant regions: 'fusiformgyrus left', 'posteriorinsula left',
        # 'leftcerebellumwhitematter', 'centraloperculum right', 'leftthalamusproper',
        if only_significant:
            # malpem significant regions
            signif_regions = ['superiorparietallobule right', 'supplementarymotorcortex left',
                              'middlecingulategyrus left', 'lateralorbitalgyrus left', 'angulargyrus left', 'subject_id', 'date_diff']

            # tractseg significant reginos
            # signif_regions = ['fronto-pontine tract left', 'corticospinal tract left',
            #                   'fronto-pontine tract right', 'parieto‐occipital pontine left', 'thalamo-premotor right', 'subject_id', 'date_diff']

            difference_md_patients = difference_md_patients[signif_regions]

            # initial_md_patients = initial_md_patients[signif_regions]

        # Getting the dataframe containing cognitive metric data and the subject_id
        desired_metrics = {metric_in_sheet: current_metric}
        metrics_df = CognitiveMetrics().isolate_metrics(False, desired_metrics)

        # Merge metrics and scan data dataframes based on their id
        multi_difference_merged = pd.merge(difference_md_patients, metrics_df, on='subject_id')
        # initial_merged = pd.merge(initial_md_patients, metrics_df, on='subject_id')

        # Add the other non brain region systematic components onto the data frame
        if consider_systematics:
            systematics = SystematicComponents()
            systematics_df = systematics.systematic_components
            multi_difference_merged = pd.merge(multi_difference_merged, systematics_df, on='subject_id')
            # initial_merged = pd.merge(initial_merged, systematics_df, on='subject_id')

        # Import other systematic components
        multi_difference_merged = multi_difference_merged.drop('subject_id', axis='columns')
        multi_difference_merged = multi_difference_merged.dropna(axis=1)

        # initial_merged = initial_merged.drop('subject_id', axis='columns')
        # initial_merged = initial_merged.dropna(axis=1)

        # singleshell data

        # Get single-shell malpem DTI data from the loaded excel file
        malpem_dti = xlsx['singleshell_malpem_FA_MD']

        # Form dataframes containing controls and patients and isolate MD data only
        md_controls, md_patients = group_metric_split(malpem_dti, 'FA')

        # Get the difference dataframe
        # Get patients with known before/after data
        md_patients_multiple = multiple_scans(md_patients)
        # # Get the extreme dataframes of the MD patients
        acute_md_patients, long_md_patients = limited_timed_dataframes(md_patients_multiple, 500)

        # Creating a dataframe for the difference in brain scan results
        acute_md_patients_temp = acute_md_patients.drop(['project_id', 'scan_date', 'scan_id', 'metric'], axis=1)
        long_md_patients_temp = long_md_patients.drop(['project_id', 'scan_date', 'scan_id', 'metric'], axis=1)
        difference_md_patients = long_md_patients_temp.subtract(acute_md_patients_temp)

        # Inserting days between scans
        difference_md_patients['date_diff'] = (long_md_patients['scan_date'] - acute_md_patients['scan_date']).dt.days

        if not ((difference_md_patients['subject_id'] == 0).all()):
            raise Exception('Difference dataframe not calculated properly')

        difference_md_patients = difference_md_patients.drop('subject_id', axis=1)
        difference_md_patients['subject_id'] = long_md_patients['subject_id']

        # Insert region names into columns and preserve the subject_id
        difference_md_patients = isolate_regions(difference_md_patients, 'malpem', True, True)

        # Only have statistically significant regions in the dataframe
        if only_significant:
            # malpem significant regions
            signif_regions = ['superiorparietallobule right', 'supplementarymotorcortex left',
                              'middlecingulategyrus left', 'lateralorbitalgyrus left', 'angulargyrus left', 'subject_id', 'date_diff']
            # tractseg significant reginos
            # signif_regions = ['fronto-pontine tract left', 'corticospinal tract left',
            #                   'fronto-pontine tract right', 'parieto‐occipital pontine left', 'thalamo-premotor right', 'subject_id', 'date_diff']

            difference_md_patients = difference_md_patients[signif_regions]


        # Getting the dataframe containing cognitive metric data and the subject_id
        desired_metrics = {metric_in_sheet: current_metric}
        metrics_df = CognitiveMetrics().isolate_metrics(False, desired_metrics)

        # Merge metrics and scan data dataframes based on their id
        single_difference_merged = pd.merge(difference_md_patients, metrics_df, on='subject_id')

        # Add the other non brain region systematic components onto the data frame
        if consider_systematics:
            systematics = SystematicComponents()
            systematics_df = systematics.systematic_components
            single_difference_merged = pd.merge(single_difference_merged, systematics_df, on='subject_id')

        # Import other systematic components
        single_difference_merged = single_difference_merged.drop('subject_id', axis='columns')
        single_difference_merged = single_difference_merged.dropna(axis=1)

        # Data harmonisation, want to create new dataframes with just the scan data
        multi_scan_only = multi_difference_merged[multi_difference_merged.columns[:-5]]

        single_scan_only = single_difference_merged[single_difference_merged.columns[:-5]]

        # Step 1: Standardize the data through taking away the mean and dividing through by the standard deviation
        single_scan_only = (single_scan_only - single_scan_only.mean()) / single_scan_only.std()
        multi_scan_only = (multi_scan_only - multi_scan_only.mean()) / multi_scan_only.std()

        # Step 2: Check for outliers and remove them
        outliers_single_shell = np.abs(single_scan_only) > 3
        outliers_multi_shell = np.abs(multi_scan_only) > 3
        single_scan_only[outliers_single_shell] = np.nan
        multi_scan_only[outliers_multi_shell] = np.nan

        # Step 3: Check for normality, if standarisation is insufficient for normality,
        # then use a logarithmic transformation
        # Flag to alert whether a transformation was necesssary
        adjusted = False
        for col in single_scan_only.columns:
            if not norm(single_scan_only[col]).rvs(len(single_scan_only[col])).shape == single_scan_only[col].shape:
                adjusted = True
                single_scan_only[col] = np.log(
                    single_scan_only[col])  # Log transformation for data that are not normally distributed
            if not norm(multi_scan_only[col]).rvs(len(multi_scan_only[col])).shape == multi_scan_only[col].shape:
                adjusted = True
                multi_scan_only[col] = np.log(
                    multi_scan_only[col])  # Log transformation for data that are not normally distributed

        if adjusted:
            print("Standardisation of data insufficient for normalisation. A log transformation was applied.")

        # Step 4: Use statistical tests to compare distributions
        ks_statistic, ks_p_value = kstest(single_scan_only.values.flatten(),
                                          multi_scan_only.values.flatten())  # Kolmogorov-Smirnov metrics_and_atlases
        shapiro_p_values = []
        anderson_p_values = []
        for col in single_scan_only.columns:
            shapiro_stat, shapiro_p_val = shapiro(single_scan_only[col].dropna() - multi_scan_only[col].dropna())
            shapiro_p_values.append(shapiro_p_val)

        plt.rcParams.update({'font.size': 14})

        # Create a bar chart of the Shapiro-Wilk p-values for each column
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(shapiro_p_values)), shapiro_p_values, color='steelblue')
        ax.set_xticks(range(len(shapiro_p_values)))
        ax.set_xticklabels(single_scan_only.columns, rotation=45)
        ax.set_xlabel('Brain region')
        ax.set_ylabel('Shapiro-Wilk p-value')
        ax.set_title('Normality metrics_and_atlases: Shapiro-Wilk')
        ax.axhline(y=0.05, color='r', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('plots/harmonisation/shapiro.png')

        # Create a histogram of the Kolmogorov-Smirnov metrics_and_atlases statistic
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ks_statistic, color='steelblue')
        ax.set_xlabel('Kolmogorov-Smirnov metrics_and_atlases statistic')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution metrics_and_atlases: Kolmogorov-Smirnov')
        ax.axvline(x=np.percentile(ks_statistic, 95), color='r', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('plots/harmonisation/ks.png')

        # Change the data accordingly such that both the single shell and multi shell data can be combined
        multi_difference_merged[multi_scan_only.columns] = multi_scan_only[multi_scan_only.columns]
        single_difference_merged[single_scan_only.columns] = single_scan_only[single_scan_only.columns]

        harmonised_data = pd.concat([multi_difference_merged, single_difference_merged], axis=0)
        harmonised_data = harmonised_data.reset_index()
        harmonised_data = harmonised_data.drop('index', axis='columns')

        # eliminate any spaces in the column names for better processing in R
        harmonised_data = harmonised_data.rename(columns=lambda x: x.replace(' ', '_'))

        with open('pickles/linear_model.pkl', 'wb') as f:
            pkl.dump(
                [harmonised_data], f
            )
        f.close()

    harmonised_data[harmonised_data.columns[:-3]] = harmonised_data[harmonised_data.columns[:-3]] + 2
    harmonised_data.to_csv(r'/Users/leondailani/Documents/Part III Project/d2.csv',index=False)
    exit()


    # # Dictionary of the regions of the brain that show the greatest change from acute to longitudinal scan
    # signif_regions = {'rightthalamusproper': 0.0008544921875, 'rightcerebralwhitematter': 0.001220703125,
    #                   'leftcerebralwhitematter': 0.0023193359375, 'rightputamen': 0.029541015625,
    #                   'fusiformgyrus left': 0.0418701171875, 'posteriorinsula left': 0.0494384765625,
    #                   'leftcerebellumwhitematter': 0.0494384765625, 'centraloperculum right': 0.0494384765625,
    #                   'leftthalamusproper': 0.0494384765625, 'rightpallidum': 0.0579833984375,
    #                   'ganteriorcingulategyrus right': 0.0784912109375, 'rightaccumbensarea': 0.0784912109375,
    #                   'superiorparietallobule right': 0.090576171875, 'frontaloperculum right': 0.10400390625,
    #                   'leftaccumbensarea': 0.10400390625}
    #
    # # Isolate these significant regions and creating a new dataframe with them and the cognitive metric to be looked at
    # signif_regions = list(signif_regions.keys())
    # signif_regions.append(current_metric)
    # significant_data_metric = initial_merged[signif_regions]

    def regression_model(df, feature_numbers, change=False, full=False):

        # For the general linear model, want to separate the data into training and testing sets to evaluate model
        # performance. Leaving 30% for testing

        brain_data_train, brain_data_test, metric_train, metric_test = train_test_split(
            df.drop(columns=[current_metric]),
            df[current_metric],
            test_size=0.3,
            random_state=42
        )

        # These lines perform feature selection using the SelectKBest method with the f_regression score function,
        # which evaluates the association between each feature and the target variable using a univariate linear
        # regression model. k=5 specifies that we want to select the top 5 most important features. The fit method
        # fits the selector to the training data, while the transform method applies the selector to both the
        # training and testing data to return a reduced feature set that includes only the most important features

        selector = SelectKBest(score_func=f_regression, k=feature_numbers)
        selector.fit(brain_data_train, metric_train)
        brain_train_selected = selector.transform(brain_data_train)
        brain_test_selected = selector.transform(brain_data_test)

        # These lines create a linear regression model and fit it to the training data using the selected features.
        reg = LinearRegression()
        reg.fit(brain_train_selected, metric_train)

        metric_train_pred = reg.predict(brain_train_selected)
        train_r2 = r2_score(metric_train, metric_train_pred)
        train_mse = mean_squared_error(metric_train, metric_train_pred)

        # These lines evaluate the performance of the model on the training set using two metrics: R-squared and mean
        # squared error. r2_score calculates the proportion of variance in the target variable that is explained by
        # the model, while mean_squared_error calculates the average squared error between the predicted and actual
        # target values

        metric_test_pred = reg.predict(brain_test_selected)
        test_r2 = r2_score(metric_test, metric_test_pred)
        test_mse = mean_squared_error(metric_test, metric_test_pred)

        # These lines evaluate the performance of the model on the testing set using the same two metrics as above.

        selected_features = selector.get_support(indices=True)
        selected_features_names = [df.columns[feature] for feature in selected_features]

        # Comparison to the null model
        null_prediction = metric_train.mean()

        null_predictions = np.full_like(metric_test, null_prediction)

        null_r2 = r2_score(metric_test, null_predictions)
        null_mse = mean_squared_error(metric_test, null_predictions)

        if change:
            return (test_r2 - null_r2)

        elif full:
            return (test_r2, test_mse, selected_features_names)

        else:
            return (test_r2)


    # # Greatest value of r^2
    # r2_values = [regression_model(initial_merged,i) for i in range(2, len(initial_merged.columns))]
    # r2_values = np.array(r2_values)
    # print(r2_values)
    # print(r2_values[r2_values.argmax()])
    # print(r2_values.argmax())

    # Greatest change from the null model
    r2_values = [regression_model(initial_merged, i, True) for i in range(2, len(initial_merged.columns))]
    r2_values = np.array(r2_values)
    print(r2_values[r2_values.argmax()])

    # Greatest value of r^2
    r2_values = [regression_model(multi_difference_merged, i) for i in range(2, len(multi_difference_merged.columns))]
    r2_values = np.array(r2_values)
    print(r2_values)
    print(r2_values[r2_values.argmax()])
    print(r2_values.argmax() + 2)

    test_r2, test_mse, selected_features_names = regression_model(multi_difference_merged, 123)

    print(selected_features_names)

    # print(len(r2_values))
    # x = np.arange(2,134)
    # print(x)
    #
    # plt.plot(x,r2_values)
    # plt.title('Number of brain regions vs r2 value of linear regression model')
    # plt.xlabel('Number of brain regions included in the model')
    # plt.ylabel('R2 value')
    # plt.savefig('model_plots/r2vsfeature number')

    # # Greatest change from the null model
    # r2_values = [regression_model(difference_merged, i, True) for i in range(2, len(difference_merged.columns))]
    # r2_values = np.array(r2_values)
    # print(r2_values[r2_values.argmax()])
    # print(r2_values.argmax() + 2)
