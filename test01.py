import pandas as pd
import os

# List of year directories containing .xpt files
year_dirs = [
    '2005-2006',
    '2007-2008',
    '2009-2010',
    '2011-2012',
    '2013-2014',
    '2015-2016',
    '2017-2020',
    '2021-2023.8'
]

# Assuming the script is run from the root directory (/home/project/yihao)
for year in year_dirs:
    dir_path = os.path.join('.', year)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.xpt'):
                filepath = os.path.join(dir_path, filename)
                try:
                    # Read the .xpt file
                    df = pd.read_sas(filepath, format='xport')
                    
                    # Convert to .csv
                    csv_path = filepath.replace('.xpt', '.csv').replace('.XPT', '.csv')  # Handle case insensitivity
                    df.to_csv(csv_path, index=False)
                    
                    # Delete the original .xpt file
                    os.remove(filepath)
                    print(f"Converted and deleted: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    else:
        print(f"Directory not found: {dir_path}")