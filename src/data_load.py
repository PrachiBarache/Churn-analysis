import pandas as pd
import os

#  data loading function
def data_loader(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    try:
        file_ext = os.path.splitext(filepath)[-1]    #can read csv or .xlsx formats

        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            if not sheet_name:
                raise ValueError("Please provide a sheet name for Excel files.")
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")

        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:            #exception if file not found
        print("no file found")





