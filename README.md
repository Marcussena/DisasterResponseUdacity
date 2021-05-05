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

### Reference
1. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html - Example of how to use sklearn classification_report
2. https://scikit-learn.org/stable/modules/compose.html - Example of how to use a pipeline
3. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html - Parameters guide and example of how to implement a Grid Search
4. https://github.com/anchorP34/Udacity-Disaster-Response-Project - This project helped me to implement the evaluate_model function in a cleaner and more efficient way
