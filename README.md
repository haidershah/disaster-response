# Disaster Response Pipeline Project

## Summary
The model classifies messages sent during natural disasters. During disasters it's a cumbersome task to manually filter out messages that are mere noise from responses where actual help is needed. To help with this problem, this project will also include a web app where an emergency worker can input a new message and get classification results in several categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

app/template/master.html - main page of web app

app/template/go.html - classification result page of web app

app/run.py - a Flask web app that runs the model on new messages and data visualization.

data/disaster_categories.csv - data to process

data/disaster_messages.csv - data to process

data/process_data.py - a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

data/InsertDatabaseName.db - database to save clean data to

models/train_classifier.py: includes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

models/classifier.pkl - saved model
