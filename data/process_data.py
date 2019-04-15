import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories datasets.

    Parameters:
        messages_filepath: Filepath of messages dataset.
        categories_filepath: Filepath of categories dataset.

    Returns:
        messages (DataFrame): Messages dataset.
        categories (DataFrame): Categories dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def clean_data(messages, categories):
    """
    Cleans data by merging messages and categories datasets, creating
    individual category columns, dropping original `categories` column,
    and dropping duplicates.

    Parameters:
        messages: Messages dataset.
        categories: Categories dataset.

    Returns:
        df (DataFrame): Cleaned and merged dataset.
    """

    # merge datasets
    df = messages.merge(categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list(row.str[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df[df.duplicated() == False]

    return df

def save_data(df, database_filename):
    """
    Saves the clean dataset into an sqlite database.

    Parameters:
        df: Dataset to save.
        database_filepath: Filepath where dataset should be saved.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

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
