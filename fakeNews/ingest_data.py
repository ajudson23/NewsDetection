# Stage 0: Data Ingestion

import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df
