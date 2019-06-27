# Disaster Response Pipeline Project

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

