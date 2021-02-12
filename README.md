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


