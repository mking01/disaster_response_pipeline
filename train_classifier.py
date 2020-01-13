# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import numpy as np
import re
#import xgboost as xgb

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report, precision_recall_fscore_support, confusion_matrix, make_scorer
#from sklearn import XGBModel

pd.set_option('max_rows', 1000)
pd.set_option('max_columns', 1000)


def load_data(database_filepath):
    """
    Purpose:  Load datasets
    
    Returns:
    X: dataframe for model features
    Y: dataframe for model prediction values
    category_names: list of strings containing category names
    """
    
    # Load data from database and save to dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)

    # Drop any null records
    df = df.dropna()

    # Create X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """
    Purpose:  Normalize and clean message text for modeling by removing stems and endings, standardizing text and formatting
    
    Input: 
    text: string (message data)

    Output: 
    Stemmed: List of cleaned strings
    """

    # Normalize by converting to lowercase and removing punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Stem word tokens
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove short words
    words = [word for word in words if len(word) > 2]
    
    return words

def __get_scores(actual, predicted):
    """
    Purpose: Calculate median F1 score for all output classifiers.  Idea adapted from GK Hayes.
    
    Inputs: 
    Actual:  Actual values
    Predicted:  Predicted values
    
    Returns:
    f1_scores:  Float.  Median F1 score.
    """
    
    # f1_scores_list = []

    # for i in range(np.shape(predicted)[1]):
    #     f1s = f1_score(np.array(actual)[:, i], predicted[:, i], average = 'micro')
    #     f1_scores_list.append(f1s)
    # f1_scores = np.median(f1_scores_list)
    # return f1_scores

    # Create blank list to hold all outcomes from loop
    all_metrics = []

    # Loop to generate stats for each column
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:, i], y_preds[:, i], normalize=True)
        precision = precision_score(y_test[:, i], y_preds[:, i], average='micro')
        recall = recall_score(y_test[:, i], y_preds[:, i], average='micro')
        f1 = f1_score(y_test[:, i], y_preds[:, i], average='micro')

    # Append all metrics to blank list
    all_metrics.append([accuracy, precision, recall, f1])

    # Convert list to dataframe
    performance_df = pd.DataFrame(np.array(all_metrics), index=category_names,
                                  columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    performance_df = performance_df.median()
    return performance_df


def build_model():
    """
    Purpose:  Construct modeling pipeline and gridsearch object.  Train and test split created under main().
    
    Inputs: None
    
    Outputs:  Gridsearch CV object that transforms data, creates the model, and finds optimal parameters
    """
    pipeline = Pipeline([
                    ('vector', CountVectorizer(tokenizer = tokenize)),
                    ('tfidf', TfidfTransformer(use_idf = True)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10)))
                    ])
    
    # Define parameters for gridsearch object
    parameters = {'clf__estimator__max_depth': [5, 15],\
                'clf__estimator__n_estimators': [5, 15],\
                'clf__estimator__max_features': [10, 15]}
                    
   # Create scorer
    f1_scorer = make_scorer(__get_scores)

    # Create gridsearch object
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = f1_scorer)
    return cv


def __get_performance_metrics(y_test, y_preds, category_names):
    """
    Purpose: Generate performance metrics for model
    
    Inputs:
    y_test:  actual data values
    y_preds:  model predicted data values
    category_names: list containing all Y category column names for each predicted category
    
    Returns: dataframe showing actual and predicted above, plus accuracy, precision, recall, and F1
    """
    # Create blank list to hold all outcomes from loop
    all_metrics = []
    
    # Loop to generate stats for each column
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:,i], y_preds[:,i], normalize = True)
        precision = precision_score(y_test[:,i], y_preds[:,i], average = 'micro')
        recall = recall_score(y_test[:,i], y_preds[:,i], average = 'micro')
        f1 = f1_score(y_test[:,i], y_preds[:,i], average = 'micro')
        
        # Append all metrics to blank list
        all_metrics.append([accuracy, precision, recall, f1])
    
    # Convert list to dataframe
    performance_df = pd.DataFrame(np.array(all_metrics), index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    return performance_df


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Purpose: Generate performance metrics for model
    
    Inputs:
    model:  model used for predicting
    X_test:  X test values
    Y_test:  actual Y test values for corresponding Xs
    category_names: list containing all Y category column names for each predicted category
    
    Returns: Nothing.  Prints dataframe showing actual and predicted above, plus accuracy, precision, recall, and F1
    """
    
    # Repeat evaluation for test set
    y_test_preds = model.predict(X_test)

    print(__get_performance_metrics(np.array(Y_test), y_test_preds, category_names))


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data and creating dataset...')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Model saved.')

    else:
        print('Please provide the filepath of the disaster messages database and the filepath of the pickle file.')


if __name__ == '__main__':
    main()