import pandas as pd
import numpy as np
import os

def clean_bhq_data(df):
    # Define valid value ranges based on the BHQ_D document
    valid_ranges = {
        'BHQ010': {1, 2, 3, 4, 5, 6, 77, 99},  # Bowel leakage (gas)
        'BHQ020': {1, 2, 3, 4, 5, 6, 77, 99},  # Bowel leakage (mucus)
        'BHQ030': {1, 2, 3, 4, 5, 6, 77, 99},  # Bowel leakage (liquid)
        'BHQ040': {1, 2, 3, 4, 5, 6, 77, 99},  # Bowel leakage (solid stool)
        'BHD050': set(range(1, 71)) | {777, 999},  # Bowel movement frequency (times per week)
        'BHQ060': {1, 2, 3, 4, 5, 6, 7, 77, 99}  # Bristol Stool Form Scale
    }

    # Convert columns to numeric and validate against valid ranges
    for col, valid_values in valid_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].isin(valid_values), np.nan)

    # Calculate Fecal Incontinence Severity Index (FISI) score
    # FISI weights from Rockwood et al, 2000 (assuming weights based on typical FISI scoring)
    fisi_weights = {
        'BHQ010': {1: 12, 2: 8, 3: 6, 4: 4, 5: 2, 6: 0},  # Gas
        'BHQ020': {1: 10, 2: 7, 3: 5, 4: 3, 5: 1, 6: 0},  # Mucus
        'BHQ030': {1: 16, 2: 12, 3: 8, 4: 6, 5: 4, 6: 0},  # Liquid stool
        'BHQ040': {1: 20, 2: 16, 3: 12, 4: 8, 5: 6, 6: 0}   # Solid stool
    }

    def calculate_fisi_score(row):
        if all(pd.notna(row[col]) and row[col] in fisi_weights[col] for col in ['BHQ010', 'BHQ020', 'BHQ030', 'BHQ040']):
            score = sum(fisi_weights[col].get(row[col], 0) for col in ['BHQ010', 'BHQ020', 'BHQ030', 'BHQ040'])
            return score
        return np.nan

    # Apply FISI score calculation
    df['fisi_score'] = df.apply(calculate_fisi_score, axis=1)

    # Count missing values for BHQ variables
    bhq_cols = [col for col in valid_ranges.keys() if col in df.columns]
    if bhq_cols:
        df['missing_bhq_count'] = df[bhq_cols].isna().sum(axis=1)
        df['all_bhq_missing'] = df['missing_bhq_count'] == len(bhq_cols)

    # Consistency check: Ensure BHD050 (bowel movement frequency) is reasonable
    if 'BHD050' in df.columns:
        df['BHD050_consistency'] = np.where(
            (df['BHD050'].notna()) & (df['BHD050'] > 70),
            'invalid_high_frequency',
            'valid'
        )

    return df

def load_data(file_path):
    # Read CSV, treating 5.397605346934028e-79 and empty fields as NaN
    df = pd.read_csv(file_path, na_values=[5.397605346934028e-79, '', np.nan])
    return df

def clean_data(file_path, output_path):
    # Load data
    df = load_data(file_path)
    
    # Clean BHQ data
    df = clean_bhq_data(df)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

def process_bhq_files(root_dir='.'):
    # Traverse all directories and process BHQ files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if (file.startswith('BHQ_') or file.startswith('P_BHQ')) and file.endswith('.csv') and '_clean' not in file:
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(subdir, file.replace('.csv', '_clean.csv'))
                print(f"Processing BHQ file: {input_file}")
                clean_data(input_file, output_file)

# Example usage
if __name__ == "__main__":
    process_bhq_files()