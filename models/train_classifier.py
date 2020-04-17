import sys
# dataframe manipulation
import pandas as pd 
import numpy as np
import re

# Database manipulation
from sqlalchemy import create_engine

# Natural Langage treatment
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, average_precision_score, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#export to pickle
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterDatabase.db')
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.drop(columns=['message', 'id', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    #text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text.lower())
    words_list = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return words_list


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    # specify parameters for grid search
    parameters = {
        'moc__estimator__min_samples_leaf':[1,3],
        'moc__estimator__min_samples_split':[2, 3]
    }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose = 4, cv = 2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    preds = model.predict(X_test)
    print(classification_report(Y_test, preds, target_names=category_names))
    print('Accuracy : ', (preds == Y_test).mean())
    print('Best parameters : ', model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train[:5000], Y_train[:5000])
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()