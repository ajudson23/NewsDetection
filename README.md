# Fake News Detection

## Object
1. Create a Natural Language Processing algorithm that is trained on real & fake news articles that can ingest outside news articles and can output real/fake predictions based on the news content.
2. Provide quantitative Metrics that can provide percentage of how much data is trustworthy


### Stage 0: Problem Definition & Data Ingestion
• Define the business problem.
• Identify and collect the dataset.
### Stage 1: Data Preparation & Data Segregation
• Process and prepare the data using techniques such as the ones mentioned in the coming slides.
• Split the data into a training set, validation set, and testing set.
### Stage 2: Model Building
• Choose an existing ML model architecture from various ML model families and then choose the hyperparameters.
### Stage 3: Model Training and Tuning
• Train the ML model using data instances in the training set. Then, tune the ML model using validat ion data instances.
### Stage 4: Candidate Model(s) Testing/Evaluation
• Use the testing dataset to measure the performance of the trained model on “unseen” in stances.
### Stage 5: Model Deployment & Performance Monitoring
• Deploy the model into production for inference.
• Continuously monitor the model’s performance. Retrain and calibrate accordingly to prevent it from becoming stale.