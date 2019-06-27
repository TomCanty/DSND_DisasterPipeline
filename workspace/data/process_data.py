import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath): 
    """ Returns df composed of the messages and categories from csv of filepaths provided for training
    
    Takes a csv (messages) with columns; id, message, original, genre, and merges it with identified 
    categories (categories) and returns it as a pandas dataframe
    
    Arguments:
    messages_filepath -- Filepath   to the csv containing the message text 
    categories_filepath -- Filepath to the csc to the categorization of 

    Returns: Pandas Dataframe
    """
    categories = pd.DataFrame.from_csv(categories_filepath)
    messages = pd.DataFrame.from_csv(messages_filepath)
    df = messages.merge(categories,left_index=True,right_index=True)

    return df



def clean_data(df):
    """ Cleans the datefram loades from the categories and messages csv
    
    One hot encodes category data and removes duplicates. Returns new dataframe

    Arguments: df - Dataframe

    Returns: Pandas Dataframe   
    """
    
    # Create column categoriy and one hot encode
    categories = df.categories.str.split(';').apply(pd.Series)
    category_colnames = list(categories.iloc[0,:].str.split("-").apply(lambda x: x[0]))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1:]))

    df = df.drop(columns='categories')
    df = df.merge(categories,left_index=True, right_index=True)

    # Remove Duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """ Saves the df to a sqlite database per filename provided
    
    Arguments: 
    df - cleaned dataframe
    database_filename - string with database filename and path

    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)
    pass  


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