# Disaster Response Pipeline Project


### Background

Following a disaster, there is often a flood of messages on social media.  The challenge with this is to parse through the messages and find where and what type help is needed urgently.  

A dataset of categorized messages was provided to help train a supervised ML model.

### Purpose

The purpose of this project is to create an ETL pipeline and a supervised ML pipeline to train a model to help categorize future disaster social media messages.  A webpage interace will be created to input new messages for categorization from the trained model and to provide data visualiztion on historical trends.

### Instructions:
1. The workspace folder is the parent directory for the working files
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

### script files and thier Requirements
1. Scripts
    1. data/process_data.py
    2. models/train_classifier.py
    3. app/run.py
2. Library Requirements:
    -   sys
    -   json
    -   plotly
    -   pandas
    -   numpy
    -   re
    -   nltk
    -   flask
    -   sklearn
    -   sqlalchemy

