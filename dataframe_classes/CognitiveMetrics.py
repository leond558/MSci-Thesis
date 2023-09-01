import pandas as pd


# A class to store the cognitive metrics spreadsheet data
class CognitiveMetrics:
    # Create a class variable with a dataframe corresponding to the sheet data
    cognitive_metrics = pd.DataFrame()

    # A class function that isolates specified desired cognitive metrics
    def isolate_metrics(self, all: bool, metrics: dict ={}):
        """
        A function that returns a dataframe with only the desired cognitive metric data.
        :param all: Include all parameters?
        :param metrics: A dictionary containing a key: value relationship relating the name of the metric on the
        columns of the sheet and their expanded/ human-readable name
        :return: New dataframe with just the specified metrics and the subject-id
        """
        # Using the entire cognitive metrics dataframe as a base
        output_df = self.cognitive_metrics


        if not all:
            # Isolating only the columns with the desired metrics
            output_df = output_df[list(metrics.keys())]
            # Rename the columns with expanded names for the metrics
            output_df = output_df.rename(columns=metrics)

            # Reinserting the subject_id column
            output_df['subject_id'] = self.cognitive_metrics['subject_id']

        output_df.columns = map(str.lower, output_df.columns)

        # Drop rows with Nan values
        output_df = output_df.dropna()

        # Convert subject_id column to be an integer so that it matches with the scan data sheet
        output_df['subject_id'] = output_df['subject_id'].astype(int)

        return output_df

    def __init__(self):
        # Read data from the Excel file and load it up
        file_path = r"/Users/leondailani/Documents/Part III Project/cognitive_metrics.xlsx"
        self.cognitive_metrics = pd.read_excel(file_path, sheet_name='Sheet1')
        # Renaming the subject_id column so that it matches the scan_data sheet
        self.cognitive_metrics = self.cognitive_metrics.rename(columns={'WBIC RIS': 'subject_id'})
