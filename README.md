# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Code files and description
1. ETL pipeline preparation.ipynb: Jupyter notebook containing the workflow for extract the data from csv file, merge and clean (transform) the data and export the resulting data into a Sqlite database (load) 
2. process_data.py: Python code created from ETL pipeline prepration.ipynb that executes a ETL pipeline from any datasets specified by the user.
3. ML pipeline preparetion.ipynb: Jupyter notebook containing the analysis performed to create train_classifier.py. It uses the database created by process_data.py to train and optimize a ML model using pipeline and Grid Search. In the end, the model is exported as a pickle file.
4. train_classifier.py: This code was created from ML pipeline preparetion.ipynb to perform the analysis, training and tuning for any database specified by the user for categorizing messages regarding 36 categories of information.
5. run.py: Flask app to implement the ML model created as a web app and display some visualizations about the training datasets. The web app receives a message from the user as input and classify the message regarding 36 possible categories.

### Reference
Follows a list of articles, web pages and github projects that I used as reference, insight and troubleshooting
1. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html - Example of how to use sklearn classification_report
2. https://scikit-learn.org/stable/modules/compose.html - Example of how to use a pipeline
3. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html - Parameters guide and example of how to implement a Grid Search
4. https://github.com/anchorP34/Udacity-Disaster-Response-Project - This project helped me to implement the evaluate_model function in a cleaner and more efficient way
5. https://knowledge.udacity.com/questions/573934 - Knowledge question that helped me to check the page when I run run.py
6. https://github.com/Kusainov/udacity-disaster-response/blob/master/ML%20Pipeline%20Preparation.ipynb - This project helped me with the parameters settings of the Grid Search since my first attempts were taking too long to fit the data.
