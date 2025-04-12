''' Stage 4: Final Model Evaluation

Testing model on unseen data to evaluate how the model performs
'''

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_on_test_data(model, test_df):
    ''' Evaluating the model on the test set '''
    X_test = test_df['cleaned_text']
    y_test = test_df['label']
    print('\tEvaluating on TEST set...')
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print_test_data_result(acc, y_test, y_pred)


def print_test_data_result(acc, y_test, y_pred):
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))