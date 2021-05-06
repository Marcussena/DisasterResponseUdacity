# Disaster Response Pipeline Project

### Motivation
This project is part of the Udacity Data Scientist Nanodegree. The objective is analyze two datasets about disasters provided by Figure Eight. From the datasets, a ETL workflow was implemented to merge and clean the data contained in the datasets and save them in a sqlite database. Then the data was splitted in training and test data and ML model was built using a pipeline and hyperparameter tuning with GridSearch. Lastly, a web app was created showing some visualizations about the training dataset where the user can write a message related to a disaster and the app will classify it according to 36 categories. The app can be really useful for people and organizations since the classification of the message helps to anticipate the aid needed by knowing what the disaster is related with (flood, shelter, security etc.).

### Instalations
The project used python 3 and the following packaged were installed:

1. NumPy
2. Pandas
3. Json
4. Plotly
5. Nltk
6. Flask
7. Sklearn
8. Sqlalchemy
9. Sys
10. Re
11. Pickle
12. Sqlite3

### Instructions and observations:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. The 'visuals' folder contains the visualizations on the web app.

5. The models folder contains a zipped file of the classifier.pkl file created in my environment.

### Code files and description
1. ETL pipeline preparation.ipynb: Jupyter notebook containing the workflow for extract the data from csv file, merge and clean (transform) the data and export the resulting data into a Sqlite database (load) 
2. process_data.py: Python code created from ETL pipeline prepration.ipynb that executes a ETL pipeline from any datasets specified by the user.
3. ML pipeline preparetion.ipynb: Jupyter notebook containing the analysis performed to create train_classifier.py. It uses the database created by process_data.py to train and optimize a ML model using pipeline and Grid Search. In the end, the model is exported as a pickle file.
4. train_classifier.py: This code was created from ML pipeline preparetion.ipynb to perform the analysis, training and tuning for any database specified by the user for categorizing messages regarding 36 categories of information.
5. run.py: Flask app to implement the ML model created as a web app and display some visualizations about the training datasets. The web app receives a message from the user as input and classify the message regarding 36 possible categories.
6. Templates folder: Contains two HTML files (go.html and master.html) needed to run the front-end of the app.

### Web app visualizations
![message_distribution](https://user-images.githubusercontent.com/55843199/117337259-8aea1200-ae73-11eb-9c32-07fe5b81c12e.png)
![top5_categories](https://user-images.githubusercontent.com/55843199/117337298-963d3d80-ae73-11eb-8a23-a122cb9a0f4e.png)
![category_frequency](https://user-images.githubusercontent.com/55843199/117347104-04d3c880-ae7f-11eb-927c-7c7a77eb5842.png)

### Acknowledgements
1. Udacity for providing the material and classes about the technical skills needed for the project and also the Knowledge section that I use frequently to answer my questions.
2. Figure Eight for providing the datasets of real applications.

### Reference
Follows a list of articles, web pages and github projects that I used as reference, insight and troubleshooting
1. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html - Example of how to use sklearn classification_report
2. https://scikit-learn.org/stable/modules/compose.html - Example of how to use a pipeline
3. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html - Parameters guide and example of how to implement a Grid Search
4. https://github.com/anchorP34/Udacity-Disaster-Response-Project - This project helped me to implement the evaluate_model function in a cleaner and more efficient way
5. https://knowledge.udacity.com/questions/573934 - Knowledge question that helped me to check the page when I run run.py
6. https://github.com/Kusainov/udacity-disaster-response/blob/master/ML%20Pipeline%20Preparation.ipynb - This project helped me with the parameters settings of the Grid Search since my first attempts were taking too long to fit the data.
