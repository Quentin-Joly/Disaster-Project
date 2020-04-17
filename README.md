# Disaster-Project

## Installation
* Pandas for the dataframes manipulations
* Numpy for the statistics and maths
* Sklearn for the predictions
* Matplotlib for the visualizations
* sqlalchemy for data extration
* nltk for NLP
* pickle
* flask to run the application on a web browser
* plotly to create graphs to run on flask

## How to run
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File explanation
* data/process_data.py : the ETL pipeline to load, clean and save the data in a SQLite database
* models/train_classifier.py : the ML pipeline used to load data from the database, fit, evaluate and export the model to a pickle object
* app/run.py : the launcher of the application
* app/templates/go.html, app/templates/master.html : templates for the web application

## Summary
The results are observable directly on the web app:
* new messages are classified into categories
* data is visualized with graphs 
