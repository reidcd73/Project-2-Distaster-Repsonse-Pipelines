import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
     Creates two pandas dataframes from loaded csv files . Then merges into a single dataframe
    
    Args:
    messages_filepath: loaction of the message.csv file 
    categories_filepath: loaction of the categories.csv file 
    -
    
    Returns:
    pandas dataframe pd 
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories ,on=['id'])
    return df


def clean_data(df):
    """
     cleans the merged dataframes df 
    
    Args:
    df: pandas dataframes 
    
    Returns:
    pandas cleaned dataframe pd 
    
    """
    categories = df['categories'].str.split(';' , expand=True)
    row = categories.head(1)
    category_colnames = row.applymap( lambda x: x[ : -2]).iloc[0].tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df , categories] , axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataframe into a sqlite database file 
    
    Args:
    df: cleaned dataframe 
    database_filename: loaction to save the sqlite database file 
        
    Returns:
    none  
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False , if_exists = 'replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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