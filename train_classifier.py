# Import all the packages needed
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

    
def load_data(database_filepath):
    """
    Function to load the database created by the ETL pipeline. The function allocates the message column into the X dataframe and the category data into y.
    It returns X and y.
    
    input: database filepath
    
    output: DataFrames X and y and a list with category names 
    
    """
    engine = create_engine('sqlite:///'+str(database_filepath))
    df = pd.read_sql ('SELECT * FROM MessagesTable', engine)
    X = df['message']
    y = df.iloc[:,4:]
    y = y.drop(['request'], axis=1)
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenization function to process the text data. As taught in class, the process consists of the following processes:
    Normalize, Tokenize, Stemming, Lemmatize and extract stop words. Returns a list of tokenized words.
    
    input: Text from the messages column
    
    output: tokenized text
    
    """
    # Normalize
    text = re.sub('\W', ' ', text.lower())
    
    stop_words = stopwords.words('english')
    
    # Tokenize, Stemming and Lemmatizing
    tokens = word_tokenize(text)
    words_stemmed = [PorterStemmer().stem(tok) for tok in tokens]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_stemmed if w not in stop_words]
    
    return lemmed


def build_model():
    
    """
    This function sets the estimators for the pipeline and the parameters for GridSearch.
    
    Input: None
    
    Output: Pipeline optimized by GridSearch
    
    """
    estimators = [('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(RandomForestClassifier()))]
    pipeline = Pipeline(estimators)
    
    parameters = {'vect__max_df': (0.75,1.0),
             'clf__estimator__n_estimators':[10,20],
             'clf__estimator__min_samples_split': [2,5]
             }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    This function prints a report with the model metrics (precision, recall, f-1 score) for each category
    
    Input: Optimized model from build_model, X_test and Y_test (from train_test_split) and the category names
    
    Output: Classification Report
    
    """
    y_pred = model.predict(X_test)
    for index, col in enumerate(y_test.columns):
        print (col, classification_report(y_test.iloc[:,index], y_pred[:,index], target_names=y_test.columns))
    


def save_model(model, model_filepath):
    
    """
    Funtion that uses pickle to export the model to a file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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
