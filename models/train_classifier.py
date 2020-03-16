import sys
# import libraries
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import sklearn
import pickle

def load_data(database_filepath):
    """
    Loads data from SQL Database 
    
    Args:
    database_filepath: SQL database file name and location 
    
    Returns:
    X : Features dataframe
    y : Target dataframe
    category_names : List of the y dataframe column labels  
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con = engine)
    
    X = df['message']
    y = df.drop(columns= ['id','message','original','genre'])
    category_names = y.columns
    return X , y , category_names


def tokenize(text):
    """
    Lemertizes and tokenises the messages 
    
    Args:
    text: message text that requires cleaning 
    
    Returns:
    clean_tokens: cleaned message text 
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    Trains model using pipeline and optermises
    
    Args:
    none
    
    Returns:
    cv: optermised model 
    """
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters =  {'tfidf__use_idf': (True, False), 
                   'clf__estimator__n_estimators': [50, 100, 200],
                   'clf__estimator__min_samples_split': [2, 4]
                   } 

    cv = GridSearchCV(pipeline, param_grid=parameters ,verbose=2, n_jobs=4)
    return cv


def evaluate_model(model, X_test, y_test , category_names):
    """
    evaluate the models performace i.e. Actual vs Predicted 
    
    Args:
    
    model: train model 
    X_test: test feature dataframe 
    y_test: test target actaul dataframe  
    catagory_names : target catagory names 
    
    Returns:
    no return 
    
    """
    
    
    
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print("Labels:", category_names)
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(y_test.values == y_pred)))
    print("Accuracy:\n", accuracy)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Save the trained model as pickle file to a user location 
    
    Args:
    model: trained model 
    model_filepath :user specified location and file name 
    
    Returns:
    none  
    
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():      
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y , category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
               
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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