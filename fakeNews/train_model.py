''' Stage 3: Training & Hyperparameter Tuning
Using the model pipeline, script will distingusishing fake & real news.
'''

from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def train_and_evaluate(model, train_df, val_df):
    ''' Train the pipelinews on training data & evaluate on validation data '''
    # extract features & labels
    X_train = train_df['cleaned_text']      # independent var: article text
    y_train = train_df['label']             # target/ouput: labels
    X_val = val_df['cleaned_text']
    y_val = val_df['label']

    # train model
    print('\tTraining the model...')
    model.fit(X_train, y_train)

    # evaluate on validation set
    print("\tEvaluating on validation set...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print_evaluation(acc, y_val, y_pred)
    print_top_words(model)
    

def print_evaluation(acc, y_val, y_pred):
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))


def print_top_words(model, n=20):
    ''' Shows the top words that have the most weight on rating fake & real '''
    vector = model.named_steps['tfidf']
    classifer = model.named_steps['clf']

    feature_names = vector.get_feature_names_out()
    coefs = classifer.coef_[0]

    top_real_words = sorted(zip(coefs, feature_names), reverse=True)[:n]
    top_fake_words = sorted(zip(coefs, feature_names))[:n]

    print("TOP words that predict REAL news:")
    for coef, word in top_real_words:
        print(f"\t{word:20s} -> {coef:.4f}")
    print("TOP words that predict FAKE news:")
    for coef, word in top_fake_words:
        print(f"\t{word:20s} -> {coef:.4f}")