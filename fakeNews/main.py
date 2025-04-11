'''
Fake News Detection
Ashley Judson
CS3820 Introduction to Artificial Intelligence

1) Create a Natural Language Processing algorithm that is trained on real & fake news articles 
   that can ingest outside news articles and can output real/fake predictions based on the news content.
2) Provide quantitative Metrics that can provide percentage of how much data is trustworthy
'''

from ingest_data import load_data

# Stage 0: Ingest the dataset
data_path = 'data/WELFake_Dataset.csv' 
# data_path = 'data/FakeNewsNet.csv'  
df = load_data(data_path)              # Loading Kaggle Dataset

# Preview to confirm it worked
print("\nmain.py:\n")
print(df.head())
