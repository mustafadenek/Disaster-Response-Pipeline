# import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_data(database_filepath):
    '''
    INPUT: 
        database_filepath - Filepath that will the file is saved.  
    OUTPUT:        
        X - Features which will be used in ML algorithm.
        Y - Outputs which will be our target value in ML algorithm
        y.keys - columns of Y
    '''
    #create connection to database
    engine = create_engine('sqlite:///data/DisasterResponse.db')       
    
    #read from SQL table
    df =  pd.read_sql_table('DisasterResponse', engine)
    
    #set X and y
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y, y.keys()

def tokenize(text):
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns tokenized, lemmatized, lowered and stripped data
    '''
    #tokenize, lemmatize and strip the data
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
  
    return clean_tokens

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features for use in GridSearchCV
        y_train: Training labels for use in GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    #create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #set parameter
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],        
        'tfidf_vect__max_df': (0.75, 1.0),
        #'clf__estimator__n_estimators': [10, 25],

    }
    #create GridSearchCV and fit it
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, Y_test, category_names):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, test set
        y_test: Label features, test set
        category_names: List of the categories 
    OUTPUT
        This method does not specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    # predict on test data
    y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))

def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        does not return anythin however, it saves the model as a pickle file.
    '''
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
