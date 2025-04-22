# Fake News Detection

## Objectives
1. Create a Natural Language Processing algorithm that is trained on real & fake news articles that can ingest outside news articles and can output real/fake predictions based on the news content.
2. Provide quantitative Metrics that can provide percentage of how much data is trustworthy

## 
### Stage 0: Problem Definition & Data Ingestion -- `ingest_data.py`
• Problem: Classify news articles as fake (0) or real (1)

• Data: Kaggle dataset with titles, source domains, retweet count, and labels. 
Kaggle Data Link: (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
### Stage 1: Data Preparation & Data Segregation -- `prepare_data.py`
• Data Prep: Clean title text (lowercase, remove punctuation, etc.)

• Split: Train (70%), Validation (15%), Test (15%)
### Stage 2: Model Building -- `build_model.py`
• TF-IDF: takes the cleaned text & converts into sparse matrix of numerical features based on word frequency & unqiueness

• Logistic Regression: Classification algorithm to learn & predict 0/1 based on the TF-IDF vectors
### Stage 3: Model Training and Tuning -- `train_model.py`
• Train the model using train and validate data from the segregated dataframe
### Stage 4: Candidate Model(s) Testing/Evaluation -- `evaluate_model.py`
• Evaluation: Accuracy, Precision/Recall, F1 Score, Confusion Matrix

##

# How to Set Up the Project
1. **Clone this repository**
   ```bash
   git clone https://github.com/ajudson23/NewsDetection.git
   cd NewsDetection

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
4. **Get Database or CSV file**
   
   Download this db from Kaggle (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). Copy the Kaggle csv file into /fakeNews/data/. You'll have to create a data folder, follow these steps:
   ```bash
   cd fakeNews/
   mkdir data
   cd data/
   
5. **Run script**
   ```bash
   cd fakeNews
   python main.py

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
