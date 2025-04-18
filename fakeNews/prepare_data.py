''' Stage 1: Data Preparation & Splitting

This file contains functions that will be cleaning and splitting the data.

 * In order to clean the data, a function modify the dataframe to be more 
   readible so an AI model can better read english/journalism diction. 
      - i.e. lowercase, remove punctuation

 * To split the data the program will be working with, a function will need to 
   split the data into training (70%), validation (15%), & Testing (15%)
'''

from sklearn.model_selection import train_test_split
import re


def clean_text(text):
    ''' Cleaning text: applying lowercase; remove URLs; remove punctuation, numbers, & extra whitespace'''
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)    
    text = re.sub(r'[^a-zA-Z\s]', '', text)       
    text = re.sub(r'\s+', ' ', text).strip()      
    return text


def split_clean_data(df):
    ''' Split & clean data '''
    # create new coln of cleaned text using text
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Split data as train (70%), validate (15%), & test (15%)
    # stratify ensures an equal distribution of (fake/real) articles
    train, temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)        # splits df into 70% test & 30% leftover (temp)
    validate, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42) # takes temp and divdes 50/50 for validate & test

    print(f"Train size: {len(train)}, validate size: {len(validate)}, Test size: {len(test)}")
    return train, validate, test