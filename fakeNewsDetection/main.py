'''
Fake News Detection
Ashley Judson
CS3820 Introduction to Artificial Intelligence

1) Create a Natural Language Processing algorithm that is trained on real & fake news articles 
   that can ingest outside news articles and can output real/fake predictions based on the news content.
2) Provide quantitative Metrics that can provide percentage of how much data is trustworthy
'''

import pandas as pd

# Load the dataset
file_path = 'FakeNewsNet.csv'  # Update if the filename is different
df = pd.read_csv(file_path)

# Preview the data
print("🔍 First 5 rows of the dataset:")
print(df.head())

# Check the shape
print(f"\n📏 Dataset shape: {df.shape}")

# View column names
print(f"\n🧾 Column names: {df.columns.tolist()}")

# Check for missing values
print("\n🧹 Missing values in each column:")
print(df.isnull().sum())

# Quick data type check
print("\n🧠 Data types:")
print(df.dtypes)

# Check label distribution
print("\n📊 Class distribution (real = 1, fake = 0):")
print(df['real'].value_counts())
