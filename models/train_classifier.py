import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['stopwords','punkt','wordnet', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
import sys

def load_data(database_filepath):
    """
    Load data from database
    
    Input: sqlite database file
    Output: X feature and y target variables
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], engine)
    
    X = df['text']
    y = df.drop(['airline', 'text_with_no_airline_tag', 'text'], axis = 1)
    
    return X, y

def tokenize(text):
    """
    Clean, tokenize  and Lemmatize text
    
    Input: text to be tokenize and lemmatize
    Output: list of clean token
    """
    
    # The following lines of code tokenize and lemmatizes words
    """
    Examples of lemmatization:
    -> plays : play
    -> corpora : corpus
    -> better : good
    """
    words = word_tokenize(text) # Tokenize text into list of words 
    
    enstopwords = stopwords.words("english")
    words = [w for w in words if not w in enstopwords] # Remove stopwords in English
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words] # Lemmatize words

    return lemmed

def build_model():
    """
    Build model
    
    Input: None
    Output: best model
    """

    pipeline = Pipeline([
                   ('vect', CountVectorizer(tokenizer=tokenize)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression())
    ])

    parameters = {
        'clf__C': [1.0],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2) 

    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model
    
    Input: model to evaluate, data input to test and label of them
    Output: classification reports of each columns (precision, recall, f1-score, accuracy)
    """

    pred = model.predict(X_test)

    print(classification_report(Y_test, pred))

def save_model(model, model_filepath):
    """
    Save model
    
    Input: model to save, filepath of the saved model 
    Output: None
    """

    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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