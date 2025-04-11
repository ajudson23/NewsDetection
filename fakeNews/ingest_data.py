''' Stage 0: Data Ingestion

Will be reading from Kaggles: FakeNewsNet.csv 
Ref: https://www.kaggle.com/datasets/algord/fake-news
'''

import pandas as pd


def load_data(filepath):
    ''' Capture the data frame from CSV file '''
    df = pd.read_csv(filepath)

    # Clean:
    if 'Unnamed: 0' in df.columns or 'index' in df.columns:     # drop index column if it exists
        df = df.drop(columns=[col for col in ['Unnamed: 0', 'index'] if col in df.columns])
    df = df.dropna(subset=['title'])    # Drop rows where title is missing

    print_dataset(df)
    return df


def print_dataset(df):
    '''Print details from the DataFrame'''
    print("First 5 rows of the dataset:")
    print(df.head())

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    # Dynamically detect the label column
    if 'real' in df.columns:
        label_column = 'real'
    elif 'label' in df.columns:
        label_column = 'label'
    else:
        print("\n⚠️ No known label column ('real' or 'label') found.")
        return

    print(f"\nClass distribution ({label_column}):")
    print(df[label_column].value_counts())
