# Disaster-Response-Pipeline


# Project Files
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
Shows data visualizations using Plotly in the web app.


