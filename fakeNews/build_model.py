''' Stage 2: Model Architecture Setup
Use TF-IDF & Logiistic Regression

 * TF-IDF: converting text into numeric values
 * Logigistic Regression model for classifying news as real or fake
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_pipeline():
    '''Creates and returns a TF-IDF + Logistic Regression model pipeline'''

    pipeline = Pipeline([
        # TFIDF will take the cleaned_text coln & converts into sparse matrix of numerical features
        # based on word frequency & unqiueness
        ('tfidf', TfidfVectorizer(
            max_features=5000,        # keep 5000 of most common words/phrases
            ngram_range=(1, 2),         # include individual words & 2 word phrases
            stop_words='english'        # remove common English stopwords (i.e. 'the' 'and', etc)
        )),
        # Classification Algo to learn to predict 0/1 based on TF-IDF vector
        ('clf', LogisticRegression(
            solver='liblinear',
            random_state=42
        ))
    ])

    return pipeline
