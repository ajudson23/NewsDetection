''' Stage 3: Training & Hyperparameter Tuning
Distingusishing fake & real news
'''

from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(model, train_df, val_df):
    ''' Train the pipelinews on training data & evaluate on validation data '''
    # extract features & labels
    X_train = train_df['cleaned_text']
    y_train = train_df['label']

    X_val = val_df['cleaned_text']
    y_val = val_df['label']

    # train model
    print('\tTraining the model...')
    model.fit(X_train, y_train)

    # Evaluate on validation set
    [print("\tEvaluating on validation set...")]
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)

    print_evaluation(acc, y_val, y_pred)
    

def print_evaluation(acc, y_val, y_pred):
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))