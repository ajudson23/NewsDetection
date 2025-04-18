''' Stage 0: Data Ingestion

Will be reading from Kaggles: FakeNewsNet.csv 

Ref: 
  https://www.kaggle.com/datasets/algord/fake-news
'''

import pandas as pd


def load_data(filepath):
    ''' Capture the data frame from CSV file '''
    df = pd.read_csv(filepath)

    # clean:
    if 'Unnamed: 0' in df.columns:                  # drop index column if it exists
        df = df.drop(columns=[col for col in ['Unnamed: 0'] if col in df.columns])
    df = df.dropna(subset=['title'])                # drop rows where title is missing
    df = df.dropna(subset=['text', 'label'])        # drop rows with missing text or label

    print_dataset(df)
    return df


def print_dataset(df):
    '''Print details from the DataFrame'''
    print("First 5 rows of the dataset:")
    print(df.head())                                # print 1st 5 entries

    print(f"\nDataset shape: {df.shape}")           # prints (# of articles rows, # coln)
    print(f"\nColumn names: {df.columns.tolist()}") # coln names: ['title', 'text', 'label']

    print("\nMissing values in each column:")
    print(df.isnull().sum())                        # get sum of missing csv elements

    print("\nData types:")
    print(df.dtypes)                                # data types: objects & int64

    # dynamically detect the label (0fake/1real) column
    if 'label' in df.columns:
        label_column = 'label'
    else:
        print("\nNo known label column ('real' or 'label') found.")
        return

    print(f"\nClass distribution ({label_column}):")    
    print(df[label_column].value_counts())          # display # of real & fake entries 
