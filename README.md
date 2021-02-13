# Disaster-Response-Pipeline

### Installation

This repository was written in HTML and Python , and requires the following Python packages: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings.

### Project Overview
This code is designed to iniate a web app which an emergency operators could exploit during a disaster (e.g. an earthquake or Tsunami), to classify a disaster text messages into several categories which then can be transmited to the responsible entity

The app built to have an ML model to categorize every message received

### Project Files
1. process_data.py

Loads the messages and categories datasets

Merges the two datasets

Cleans the data

Stores it in a SQLite database

2. train_classifier:

Loads data from the SQLite database

Splits the dataset into training and test sets

Builds a text processing and machine learning pipeline

Trains and tunes a model using GridSearchCV

Outputs results on the test set

Exports the final model as a pickle file

3. Flask Web App

shows data visualizations using Plotly in the web app.

### How to Run the Program
1. Run the following commands in the project's root directory to set up your database and model.

      -To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
      
      -To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
      
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/
