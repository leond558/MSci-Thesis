import pandas as pd


# A class to store the cognitive metrics spreadsheet data
class SystematicComponents:
    # Create a class variable with a dataframe corresponding to the sheet data
    systematic_components = pd.DataFrame()

    def __init__(self):
        # Read data from the Excel file and load it up
        file_path = r"/Users/leondailani/Documents/Part III Project/systematic_components.xlsx"
        self.systematic_components = pd.read_excel(file_path, sheet_name='Sheet2')
