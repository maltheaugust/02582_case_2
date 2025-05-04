import pandas as pd
import os
import numpy as np

# Load the dataset
def load(csv: str, cohort=None) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, csv)

    df = pd.read_csv(file_path)

    # Define column groups
    raw_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                          'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    meta_cols = ['Team_ID', 'Round', 'Phase', 'Puzzler', 'raw_data_path', 'Individual', 'original_ID', 'Cohort']

    # Fill missing values in questionnaire columns with mode
    for col in questionnaire_cols:
        if col in df.columns:
            mode = df[col].mode()
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)

        # Fill missing values in raw feature columns with mean
    for col in raw_feature_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    if cohort:
        df = df[df['Cohort']==cohort]

    raw_data = df[raw_feature_cols + questionnaire_cols + ['Phase'] + ['Puzzler']]
    # Include 'Phase' with raw data
    
    metadata = df[meta_cols]

    return raw_data, metadata


def load_and_normalize_by_individual(csv, cohort="D1_2"):
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, csv)

    df = pd.read_csv(file_path)

    # Define column groups
    raw_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                          'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    meta_cols = ['Team_ID', 'Round', 'Phase', 'Puzzler', 'raw_data_path', 'Individual', 'original_ID', 'Cohort']


    # Fill missing values in questionnaire columns with mode
    for col in questionnaire_cols:
        if col in df.columns:
            mode = df[col].mode()
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)

        # Fill missing values in raw feature columns with mean
    for col in raw_feature_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    df = df[df['Cohort']==cohort]

    # Include 'Individual' in raw_data to allow per-individual normalization
    raw_data = df[raw_feature_cols + questionnaire_cols + ['Phase', 'Puzzler', 'Individual']]
    metadata = df[meta_cols]

    individual_means = raw_data.groupby('Individual')[raw_feature_cols + questionnaire_cols].mean().reset_index()
    
    # Merge means into raw_data
    raw_data = raw_data.merge(
        individual_means, 
        on='Individual', 
        suffixes=('', '_ind_mean'))
    
    # Subtract individual mean
    features = raw_feature_cols + questionnaire_cols
    for col in features:
        raw_data[col] = raw_data[col] - raw_data[f'{col}_ind_mean']
    
    # Drop mean columns
    raw_data.drop(columns=[f'{col}_ind_mean' for col in features], inplace=True)

    # --- STEP 2: Standardize globally (unit variance) ---
    # Compute global standard deviation for each feature
    global_stds = raw_data[features].std()
    
    # Avoid division by zero (replace 0 with 1)
    global_stds.replace(0, 1, inplace=True)
    
    # Divide by global standard deviation
    raw_data[features] = raw_data[features] / global_stds

    return raw_data, metadata

if __name__ == "__main__":
    df = load("data/HR_data_2.csv")
    raw_data, metadata = df
    print("Number of NaN values in raw_data:", raw_data.isna().sum().sum())
    print("Number of NaN values in metadata:", metadata.isna().sum().sum())
    
    raw_data, metadata = load_and_normalize_by_individual("data/HR_data_2.csv")
    # print(raw_data)
    
