import pandas as pd
import os

# Load the dataset
def load(csv: str) -> pd.DataFrame: 
    
    # parent folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # absolute path to the CSV file
    file_path = os.path.join(script_dir, csv)

    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    df = load("data/HR_data_2.csv")
    print(df.head())