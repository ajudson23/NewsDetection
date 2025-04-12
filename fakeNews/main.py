'''
Fake News Detection
Ashley Judson
CS3820 Introduction to Artificial Intelligence

1) Create a Natural Language Processing algorithm that is trained on real & fake news articles 
   that can ingest outside news articles and can output real/fake predictions based on the news content.
2) Provide quantitative Metrics that can provide percentage of how much data is trustworthy
'''

from ingest_data import load_data
from prepare_data import split_clean_data
from build_model import build_pipeline

# Stage 0: Ingest the dataset
print("\n********************** STAGE 0 **********************\n")
data_path = 'data/WELFake_Dataset.csv' 
df = load_data(data_path)              # Loading Kaggle Dataset

# Stage 1: Clean & Split
print("\n********************** STAGE 1 **********************\n")
train_df, val_df, test_df = split_clean_data(df)
print(df.head())

# Stage 2: Model Architecture Setup & Build Pipeline
print("\n********************** STAGE 2 **********************\n")
model = build_pipeline()
print(model)

# Stage 3:
print("\n********************** STAGE 3 **********************\n")