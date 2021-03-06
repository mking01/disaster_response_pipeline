# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

pd.set_option('max_rows', 1000)
pd.set_option('max_columns', 1000)

def load_data(messages_filepath, categories_filepath):
    '''
    :param messages_filepath: file path for messages.csv data
    :param categories_filepath: file path for categories.csv data
    :return: dataframe with messages and categories data
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''

    :param df: dataframe with messages and categories data
    :return: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand = True))
    
    # select the first row of the categories dataframe
    col_names = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    col_names = [x[:-2] for x in col_names]

    # rename the columns of `categories`
    categories.columns = col_names
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    # joining on index as the two dataframes have no common columns
    df = df.join(categories, how='outer')
    
    # drop duplicates
    df = df.drop_duplicates(subset = ['id'],keep = 'first')
    
    return df
    
    
def save_data(df, database_filename):
    '''

    :param df: cleaned dataframe
    :param database_filename: name of database
    :return:
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists = 'replace')


def main():
    '''

    :return: none, class structure that shows how the script processes
    '''
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