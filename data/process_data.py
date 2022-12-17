import sys
import re
import pandas as pd 
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def load_data(tweets_filepath):
    """
    Load data from files
    
    Input: Tweets datasets files
    Output: 2 dataframes respectively
    """

    # load datasets
    df = pd.read_csv(tweets_filepath)    
    
    return df

def clean_data(df):
    """
    Clean the dataframe
    
    Input: Dataframe read in csv
    Output: Dataframe after cleaning
    """
    # List of stop words in English
    enstopwords = stopwords.words("english") 
    
    # Columns text with no airline tag only keep characters @ a-z A-Z 
    df['text_with_no_airline_tag'] = df['text'].apply(lambda str: re.sub('[^@a-zA-Z]',' ',str))
    
    # Remove airline tag like @united from the string
    df['text_with_no_airline_tag'] = df['text_with_no_airline_tag'].apply(lambda str: re.sub('@[a-zA-Z]+',' ',str))

    # Define list of columns to remove stop words and convert to lowercase
    cols = ['airline', 'text_with_no_airline_tag', 'text']

    # Remove stop words and convert to lowercase
    for col in cols: 
        df[col].apply(lambda str: str.lower())
        df[col].apply(lambda str: [word for word in str.split() if not word in enstopwords])

    # Join list of words with space 
    df['text'].apply(lambda str: ' '.join(str))
    df['text_with_no_airline_tag'].apply(lambda str: ' '.join(str))

    # Join list of words with '' 
    df['airline'].apply(lambda str: ''.join(str))

    # Convert label value from string to numeric
    map_label_dict = {'positive':1, 'negative':-1, 'neutral':0}
    df = df.replace({'airline_sentiment':map_label_dict})

    # Filter only 4 columns 
    df = df[['airline', 'text_with_no_airline_tag', 'text', 'airline_sentiment']]

    return df

def save_data(df, database_filename):
    """
    Save dataframe into sqlite database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('TweetsETL', engine, if_exists = 'replace', index=False)

def main():
    if len(sys.argv) == 3:

        tweets_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    TWEETS: {}\n    '
              .format(tweets_filepath))
        df = load_data(tweets_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
   


