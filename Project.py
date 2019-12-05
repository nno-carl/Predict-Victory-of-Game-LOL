# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:40:36 2019

@author: cyx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Binarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from skrules import SkopeRules
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.externals import joblib

#build model and test on test set
def testdataset(algorithm, x_train, y_train, x_test, y_test, name):
    model = algorithm.fit(x_train, y_train)#train on training set
    y_score = model.predict_proba(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])#calculate fpr,tpr and threshold on test set
    roc_auc = auc(fpr, tpr)#calculate area of roc curve on test set
    
    Draw(fpr, tpr, roc_auc, name)#draw roc curve of test set
    
    #compute accuracy, precisioin and recall on test set
    predict = model.predict(x_test)
    predict_accuracy = model.score(x_test, y_test)
    predict_precision = precision_score(y_test, predict)
    predict_recall = recall_score(y_test, predict)
    
    print(f"{name} predict accuracy: {predict_accuracy:.6f}")
    print(f"{name} predict roc: {roc_auc:.6f}")
    print(f"{name} predict precision: {predict_precision:.6f}")
    print(f"{name} predict recall: {predict_recall:.6f}")
    print("")
    
    #joblib.dump(model, f"{name}_model.m")
    
#build rule based model and test on test set. methods of rule based model is different from others
def rulebased(algorithm, x_train, y_train, x_test, y_test, name):
    model = algorithm.fit(x_train, y_train)#train on training set
    y_score = model.decision_function(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)#calculate fpr,tpr and threshold on test set
    roc_auc = auc(fpr, tpr)#calculate area of roc curve on test set
    
    Draw(fpr, tpr, roc_auc, name)#draw roc curve of test set
    
    #compute accuracy, precisioin and recall on test set
    predict = model.predict(x_test)
    predict_accuracy = accuracy_score(y_test, predict)
    predict_precision = precision_score(y_test, predict)
    predict_recall = recall_score(y_test, predict)
    
    print(f"{name} predict accuracy: {predict_accuracy:.6f}")
    print(f"{name} predict roc: {roc_auc:.6f}")
    print(f"{name} predict precision: {predict_precision:.6f}")
    print(f"{name} predict recall: {predict_recall:.6f}")
    print("")
    
    #joblib.dump(model, f"{name}_model.m")
    
#10-fold cross-validation
def crossvalidation(algorithm, x, y, name):
    #calculate accuracy, area of roc curve, precision and recall of 10-fold cross-validation
    accuracy = cross_val_score(algorithm, x, y, cv = 10, scoring = 'accuracy', n_jobs=-1)
    roc = cross_val_score(algorithm, x, y, cv = 10, scoring = 'roc_auc', n_jobs=-1).mean()
    precision = cross_val_score(algorithm, x, y, cv = 10, scoring = 'precision', n_jobs=-1).mean()
    recall = cross_val_score(algorithm, x, y, cv = 10, scoring = 'recall', n_jobs=-1).mean()
    
    print(f"{name} accuracy: {accuracy.mean():.6f}")
    print(f"{name} roc: {roc:.6f}")
    print(f"{name} precision: {precision:.6f}")
    print(f"{name} recall: {recall:.6f}")
    print(f"{name} 10-fold accuracy: ")
    print(accuracy)
    print("")
    
    return accuracy
    
#draw roc curve for each model
def Draw(fpr, tpr, roc_auc, name):
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'{name} ROC curve (area = {roc_auc:.6f})') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC CURVE')
    plt.legend(loc="lower right")
    plt.show()

#t-test, compute differece, average of difference, standard deviation of difference,
#t-score and p-value between two algorithms
def Ttest(accuracy1, accuracy2, name1, name2):
    dif = accuracy1 - accuracy2
    avg = dif.mean()
    std = dif.std()
    t, p = ttest_ind(accuracy1, accuracy2)
    print(f"{name1}-{name2} difference:")
    print(dif)
    print(f"{name1}-{name2}: average = {avg:.6f} std = {std:.6f}")
    print(f"t-test {name1}-{name2}: t-score = {t:.6f} p-value = {p:.6f}")
    print(t,p)
    print("")

if __name__ == "__main__":
    #read dataset and divide dataset into feature datset and target datset
    dataset = pd.read_csv("./dataset/games.csv")
    data = dataset.drop("winner", axis = 1)
    target = dataset[['winner']]
    
    #select numeric features and categorical features, remove 'gameID' 'creationTime' and 'seasonID'
    numeric_feature = ['gameDuration', 't1_towerKills', 't1_inhibitorKills', 't1_baronKills',
                                       't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills', 't2_inhibitorKills',
                                       't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
    categorical_feature = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon','firstRiftHerald',
                                        't1_champ1id', 't1_champ1_sum1', 't1_champ1_sum2',
                                        't1_champ2id', 't1_champ2_sum1', 't1_champ2_sum2',
                                        't1_champ3id', 't1_champ3_sum1', 't1_champ3_sum2',
                                        't1_champ4id', 't1_champ4_sum1', 't1_champ4_sum2',
                                        't1_champ5id', 't1_champ5_sum1', 't1_champ5_sum2',
                                        't1_ban1', 't1_ban2', 't1_ban3', 't1_ban4', 't1_ban5',
                                        't2_champ1id', 't2_champ1_sum1', 't2_champ1_sum2',
                                        't2_champ2id', 't2_champ2_sum1', 't2_champ2_sum2',
                                        't2_champ3id', 't2_champ3_sum1', 't2_champ3_sum2',
                                        't2_champ4id', 't2_champ4_sum1', 't2_champ4_sum2',
                                        't2_champ5id', 't2_champ5_sum1', 't2_champ5_sum2',
                                        't2_ban1', 't2_ban2', 't2_ban3', 't2_ban4', 't2_ban5']
    
    #binarize the target feature
    binarizer = Binarizer(threshold = 1.5)
    target = binarizer.fit_transform(target)
    y = np.ravel(target)
    
    #count the number of two classes in target feature to know the balance of dataset
    print('number of target = 0 (team 1 win): %d' % np.sum(target == 0))
    print('number of target = 1 (team 2 win): %d' % np.sum(target == 1))
    print("")
    
    #normalize the numeric features, make their range [0, 1]
    scaler = MinMaxScaler()
    data1 = scaler.fit_transform(data[numeric_feature])
    
    #calibrate the categorical features, 
    encoder = OneHotEncoder(categories = 'auto', sparse = False)
    data2 = encoder.fit_transform(data[categorical_feature])
    
    #merge preprocessed features
    x = np.append(data1, data2, axis = 1)
    print('number of features after preprocessing: %d' % len(x[0]))
    print("")
    
    #use extra trees for feature selection
    clf = ExtraTreesClassifier(n_jobs=-1, random_state=0)
    clf = clf.fit(x,y)
    model = SelectFromModel(clf, prefit=True)
    x = model.transform(x)
    print('number of features after feature selection: %d' % len(x[0]))
    print("")
    
    #calculuate feature importance and output top 30 features with high importance
    categorical_name = encoder.get_feature_names(categorical_feature)
    feature_name = np.append(numeric_feature, categorical_name)
    feature_importance = clf.feature_importances_
    print('average feature importance: %f' % feature_importance.mean())
    print("")
    importance = dict(zip(feature_name, feature_importance))
    importance_sorted = sorted(importance.items(), key = lambda x: x[1], reverse=True)
    print("top 30 features with high importance:")
    print(importance_sorted[0:30])
    print("")

    #dimensionality reduction by PCA
    x = PCA(n_components = 60).fit_transform(x)

    #divide feature dataset and target dataset into training set and test set respectively
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
    
    #set up classifiers
    knn = KNeighborsClassifier(n_jobs=-1)
    dt = DecisionTreeClassifier(random_state=0)
    nb = GaussianNB()
    rb = SkopeRules(n_jobs=-1, random_state=0)
    lr = LogisticRegression(n_jobs=-1, random_state=0)
    rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    
    #build knn model, test on test set and 10-fold cross-validation
    testdataset(knn, x_train, y_train, x_test, y_test, "KNN")
    knn_accuracy = crossvalidation(knn, x, y, "KNN")
    
    #build decision tree model, test on test set and 10-fold cross-validation
    testdataset(dt, x_train, y_train, x_test, y_test, "Decision Tree")
    dt_accuracy = crossvalidation(dt, x, y, "Decision Tree")
    
    #build naive bayes model, test on test set and 10-fold cross-validation
    testdataset(nb, x_train, y_train, x_test, y_test, "Naive Bayes")
    nb_accuracy = crossvalidation(nb, x, y, "Naive Bayes")
    
    #build rule based model, test on test set and 10-fold cross-validation
    rulebased(rb, x_train, y_train, x_test, y_test, "Rule-based")
    rb_accuracy = crossvalidation(rb, x, y, "Rule-based")
    
    #build logistic regression model, test on test set and 10-fold cross-validation
    testdataset(lr, x_train, y_train, x_test, y_test, "Logistic Regression")
    lr_accuracy = crossvalidation(lr, x, y, "Logistic Regression")
    
    #build random forest model, test on test set and 10-fold cross-validation
    testdataset(rf, x_train, y_train, x_test, y_test, "Random Forest")
    rf_accuracy = crossvalidation(rf, x, y, "Random Forest")
    
    #t-test among algorithms
    Ttest(knn_accuracy, dt_accuracy, "knn", "decision tree")
    Ttest(knn_accuracy, nb_accuracy, "knn", "naive bayes")
    Ttest(knn_accuracy, rb_accuracy, "knn", "rule based")
    Ttest(knn_accuracy, lr_accuracy, "knn", "logistic regression")
    Ttest(knn_accuracy, rf_accuracy, "knn", "random forest")
    Ttest(dt_accuracy, nb_accuracy, "decision tree", "naive bayes")
    Ttest(dt_accuracy, rb_accuracy, "decision tree", "rule based")
    Ttest(dt_accuracy, lr_accuracy, "decision tree", "logistic regression")
    Ttest(dt_accuracy, rf_accuracy, "decision tree", "random forest")
    Ttest(nb_accuracy, rb_accuracy, "naive bayes", "rule based")
    Ttest(nb_accuracy, lr_accuracy, "naive bayes", "logistic regression")
    Ttest(nb_accuracy, rf_accuracy, "naive bayes", "random forest")
    Ttest(rb_accuracy, lr_accuracy, "rule based", "logistic regression")
    Ttest(rb_accuracy, rf_accuracy, "rule based", "random forest")
    Ttest(lr_accuracy, rf_accuracy, "logistic regression", "random forest")
