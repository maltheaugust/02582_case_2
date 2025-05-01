import pandas as pd
import os

# Load the dataset
def load(csv: str) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, csv)

    df = pd.read_csv(file_path)

    # Define column groups
    raw_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                          'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    meta_cols = ['Team_ID', 'Round', 'Puzzler', 'raw_data_path', 'Individual', 'original_ID', 'Cohort']

    # Include 'Phase' with raw data
    raw_data = df[raw_feature_cols + questionnaire_cols + ['Phase'] + ['Puzzler']]
    metadata = df[meta_cols]

    return raw_data, metadata


if __name__ == "__main__":
    df = load("data/HR_data_2.csv")
    print(df.head())