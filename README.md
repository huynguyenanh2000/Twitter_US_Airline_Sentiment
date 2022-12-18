# **Disaster Response Pipeline Project**


## **Table of contents**

- [Environment Setup](#environment-setup)
- [Project Descriptions](#project-descriptions)
- [File Structure](#file-structure)
- [Usage](#usage)


## **Environment Setup**

**Environment**
- OS: Windows 11

- Interpreter: Visual Studio Code

- Python version: Python 3.7

**Libraries**
- Install all packages using requirements.txt file using `pip install -r requirements.txt` command.

**Link to GitHub Repository**

`https://github.com/huynguyenanh2000/Twitter_US_Airline_Sentiment.git`


## **Project Descriptions**

This project is a part of Udacity Data Scientist Nanodegree program. 

### **Project Overview**
This project takes feedback from customers using flight services of airlines in the US and classifies it as negative, positive or neutral.

### **Problem Statement**
The purpose of this project is to train and evaluate the machine learning model to classify text as customer reviews about flight services with the best accuracy. 

### **Metrics** ####
Because this is a classification problem, i choose accuracy metrics. The metrics is calculated with classification report on test set.

## **File Structure**

~~~~~~~
twister_us_airline_sentiment
    |-- app
        |-- templates
                |-- go.html
                |-- master.html
        |-- run.py
    |-- data
        |-- TweetsETL.db
        |-- Tweets.csv
        |-- process_data.py
    |-- models
        |-- TweetsModel.pkl
        |-- train_classifier.py
    |-- notebook
        |-- tweets.ipynb
    |-- README
    |-- requirements.txt
~~~~~~~


## **Usage**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/Tweets.csv  data/TweetsETL.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/TweetsETL.db models/TweetsModel.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage