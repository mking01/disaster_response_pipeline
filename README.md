# Disaster Response Pipeline
Analyze disaster data via building ETL and ML pipelines using data engineering skills

# Installation
This project uses the following packages: pandas, nltk, sklearn, numpy, flask, json, plotly, re, pickle, and sqlalchemy.
  
# Overview
This project builds a model to classify responses submitted after natural disasters.  The goal of the model is to correctly predict the type of message using only text.  Correct classification allows responders to prioritize responses and correctly and efficiently allocate resources at times when they are needed most.

The project has three main parts:
1.  An ETL pipeline to process data
2.  A machine learning pipeline to build the model and classify messages
3.  A web app to easily visualize and gain insights regarding messages received


# Repository File Descriptions
- process_data.py:  ETL pipeline to process raw data for modeling.  It accepts .csv files as inputs and outputs a SQLlite database.
- train_classifier.py:  Modeling pipeline to build model and create predictions for messages.  It uses the SQLlite database created from process_data.py and outputs a pickled model.
- Disaster Response - ETL Pipeline.ipynb:  Contains the preliminary code used for analysis and developing the process_data.py pipeline
- ML Pipeline Preparation.ipynb:  Contains the preliminary code used for analysis and developing the train_classifier.py pipeline
- data:  Folder containing all data used for the project.  It contains two files, messages data and categories data.
- app:  Folder containing:
  1. run.py: Python file that creates and runs the web app
  2.  web_app_templates.tar.gz:  Template files used by run.py to create the web app

# Instructions to Run
1.  Run process_data.py
 - From the current working directory, run the following command: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2.  Run train_classifier.py
  - In the current working directory, create a 'models' folder.  Save train_classifier.py there.
  - Run the following command: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3.  Change directories to the app folder
  - Run `python run.py`
  - Go to http://0.0.0.0:3001/ to see the app

# Web App
Alternate URL:  https://view6914b2f4-3001.udacity-student-workspaces.com/

Screenshot 1:
<img width="1355" alt="Screen Shot 2020-01-10 at 8 54 53 AM" src="https://user-images.githubusercontent.com/59142310/72162085-e8a54180-3386-11ea-952c-313edcac278e.png">

Screenshot 2:
<img width="1283" alt="Screen Shot 2020-01-10 at 8 55 57 AM" src="https://user-images.githubusercontent.com/59142310/72162160-0b375a80-3387-11ea-9692-d17d8f536c9c.png">


# Acknowledgements
This project was completed as part of [Udacity's Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
