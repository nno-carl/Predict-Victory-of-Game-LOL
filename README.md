# Predict-Victory-of-Game-LOL
This is the project for course Machine Learning.  
The project is to use machine learning algorithms to predict the outcome of the game League of Legends and discover some important features that have great impact on the outcome of the game.  

The work of the project: check the balance of the dataset, preprocess the features, feature selection, dimensionality reduction, use different algorithms to build models, test models on the test set, 10-fold cross-validation and statistical test.  

The dataset contains 51490 rows and 61 features. The resource of the dataset: https://www.kaggle.com/datasnaek/league-of-legends  

The models used for prediction in the project: k-nearest neighbor, decision tree, naive bayes, rule-based, logistic regression and random forest.  
The algorithm used for feature selection: extra trees  
The algorithm used for dimensionality reduction: PCA  
The method used for statistical test: t-test  

After test on the test set and 10-fold cross-validation, the random forest model is the best.

The 'project.py' includes all code of the project.  
The dataset is in the 'games.csv.zip' and the folder 'dataset'  
The results are in 'Results.txt'. The pictures of roc curve on the test set are in the folder 'ROC'  
The models (except k-nn model, it exceeded upload file size limit) trained in the project are in the folder 'Models'  

Tools:  
Python Scikit-learn Numpy Pandas Scipy matplotlib
