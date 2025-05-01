import pandas as pd
import os

# Load the dataset
def load(csv: str) -> tuple[pd.DataFrame, pd.DataFrame]: 
    
    # parent folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # absolute path to the CSV file
    file_path = os.path.join(script_dir, csv)

    df = pd.read_csv(file_path)
    
    raw_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    meta_cols = ['Team_ID', 'Round', 'Phase', 'Puzzler', 'raw_data_path', 'Individual', 'original_ID', 'Cohort']
    questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    

    # split into raw features + questionnaire and metadata
    raw_data = df[raw_feature_cols + questionnaire_cols]
    metadata = df[meta_cols]

    return raw_data, metadata


if __name__ == "__main__":
    raw_data, metadata = load("data/HR_data_2.csv")
    print("Raw data:")
    print(raw_data.head())
    print("\nMetadata:")
    print(metadata.head())