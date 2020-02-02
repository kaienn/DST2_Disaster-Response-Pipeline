# DST2_Disaster-Pipeline

## Table of Contents
1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, Acknowledgements](#other)

## Installation <a name="installation"></a>
The codes were written in Python 3, and require the following Python packages: sys, json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, numpy, re, pickle.
* **Required libraries**
  - nltk 3.2.5
  - numpy 1.12.1
  - pandas 0.23.3
  - scikit-learn 0.19.1
  - sqlalchemy 1.2.18
  - packages need to be installed for nltk: punkt, wordnet, stopwords

## Project Overview <a name="overview"></a>
In this project, I am building a model for an API that classifies disaster messages with disaster data from Figure Eight. The dataset cotains real messages that were sent during disaster events. I created a maching learning pipeline to categorize these events so that these messages can be sent to an appropriate disaster relief agency.

It also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## File Descriptions <a name="files"></a>

* **ETL Pipeline Preparation.ipynb**: The code is for ETL Pipeline and store clean data in sql db.
* **ML Pipeline Preparation.ipynb**: The code is for ML Pipeline and prediction of messages.
* **data**: This folder contains sample messages and categories datasets in csv format.
  - disaster_categories.csv  # data to process
  - disaster_messages.csv  # data to process
  * **process_data.py**: This code takes as its input csv files containing message data. The data undergoes ETL Pipeline and get prepared to run Machine Learning Model. 
  - InsertDatabaseName.db   # database to save clean data to
* **app**: This folder contains all of the files necessary to run and render the web app.
  - template
  - master.html  # main page of web app
  - go.html  # classification result page of web app
  - run.py  # Flask file that runs app 
* **models**: This folder contains file to run the classification and classification.pkl
  * **train_classifier.py**: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
  - classifier.pkl  # saved model 

## Instructions <a name="instructions"></a>
### ***Run process_data.py***
1. Save the data folder in the current working directory and process_data.py in the data folder.
2. From the current working directory, run the following command:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run train_classifier.py***
1. In the current working directory, create a folder called 'models' and save train_classifier.py in this.
2. From the current working directory, run the following command:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### ***Run the web app***
1. Save the app folder in the current working directory.
2. Run the following command in the app directory (with app dir as root, cd):
    `python run.py`
3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements <a name="other"></a>
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) with disaster data provided by Figure Eight.
