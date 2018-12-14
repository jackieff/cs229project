#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 06:44:36 2018

@author: jackieff, vkozlow, margalan

CS 229 Final Project code
This is for the final version, everything done previously for the milestone is
included in the "milestone_code.py" file.

We ran the code bit by bit, so many variable names are reused.
"""
#importing libraries
import os
import conda
import sys
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import rcParams
from imblearn.over_sampling import SMOTE
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold

#changing default parameters for plotting with matplotlib
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams.update({'font.size': 14})


def plot_confusion_matrix(cm, classes, title,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Printing and plotting confusion matrix (cm) with optional
    normalization
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    #plotting routine adapted from matplotlib example

df = pd.read_csv('training_set_values.csv') #reading training set features

"""
Training set features in df dataframe contain two out of the three outputs we
are predicting as output with our algorithms (water quality and water quantity).

For each of the three output predictions, we are using the other two as input
features (i.e., water quality and functionality of pump are both predictors for
water quantity, etc.)

"""
labels = pd.read_csv('training_set_labels.csv') #reading training set labels for functionality of pumps

df['labels']=labels['status_group'] #adding labels to the dataframe

#dropping variables that were deemed repetitive/not useful
df =df.drop(columns=['id','wpt_name', 'num_private', 'subvillage', 'region', 'district_code', 'lga', 'ward', 'recorded_by', 'scheme_name', 'extraction_type_group', 'payment', 'water_quality', 'quantity', 'source_type', 'waterpoint_type_group'])

#creating a new variable of "age" from the construction_year and date_recorded raw features
df['date_recorded'] = df['date_recorded'].astype(str).str[:-6] #getting the year of the date recorded
df['age']=df['date_recorded'].astype(int)-df['construction_year']
ind = df['age']>2000
df['age'][ind]=np.nan #if there is no construction date set the age to NaN

df = df.drop(columns = ['date_recorded', 'construction_year'])

#editing variable from hundreds of categories to funder = 1 if village/villagers, 0 otherwise
df['funder']=df['funder'].fillna('0')
df['funder'] = df['funder'].str.contains('|'.join(['Village','village'])).astype(int) #binary


#places with 0 longitude and 2e-08 latitude had missing data for those 2 features
ind = df['longitude']==0.0
df['longitude'][ind]=np.nan

ind = df['latitude']==-2e-08
df['latitude'][ind]=np.nan

#similar to funder, reducint from hundreds of categories to installer = 1 if village/villagers, 0 otherwise
df['installer']=df['installer'].fillna('0')
df['installer'] = df['installer'].str.contains('|'.join(['Village','village'])).astype(int) #binary

#from T/F to 0/1
ind = df.notna()['public_meeting']
df['public_meeting'][ind]=df['public_meeting'][ind].astype(int)

ind = df.notna()['permit']
df['permit'][ind]=df['permit'][ind].astype(int)


#Here begins One Hot Encoding (OHE) for all remaining features
new_basin = np.asarray(df['basin']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_basin)
for num,i in enumerate(np.unique(df['basin'])):
    df['basin_'+str(i)] = a[:,num]
df = df.drop(columns = 'basin')

new_region = np.asarray(df['region_code']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_region)
for num,i in enumerate(np.unique(df['region_code'])):
    df['region_'+str(i)] = a[:,num]
df = df.drop(columns = 'region_code')

df['scheme_management'] = df['scheme_management'].fillna('znan')
new_scheme = np.asarray(df['scheme_management']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_scheme)
for num,i in enumerate(np.unique(df['scheme_management'])):
    df['scheme_'+str(i)] = a[:,num]
df = df.drop(columns = ['scheme_znan', 'scheme_management'])

new_extype = np.asarray(df['extraction_type']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_extype)
for num,i in enumerate(np.unique(df['extraction_type'])):
    df['exttype_'+str(i)] = a[:,num]
df = df.drop(columns = 'extraction_type')


new_extypeclass = np.asarray(df['extraction_type_class']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_extypeclass)
for num,i in enumerate(np.unique(df['extraction_type_class'])):
    df['exttypeclass_'+str(i)] = a[:,num]
df = df.drop(columns = 'extraction_type_class')


new_mgmt = np.asarray(df['management']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False, categories='auto')
a = enc.fit_transform(new_mgmt)
for num,i in enumerate(np.unique(df['management'])):
    df['mgmt_'+str(i)] = a[:,num]
df = df.drop(columns = 'management')


new_mgmtgp = np.asarray(df['management_group']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_mgmtgp)
for num,i in enumerate(np.unique(df['management_group'])):
    df['mgmtgp_'+str(i)] = a[:,num]
df = df.drop(columns = 'management_group')


new_paytype = np.asarray(df['payment_type']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_paytype)
for num,i in enumerate(np.unique(df['payment_type'])):
    df['paytype_'+str(i)] = a[:,num]
df = df.drop(columns = 'payment_type')


new_source = np.asarray(df['source']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_source)
for num,i in enumerate(np.unique(df['source'])):
    df['source_'+str(i)] = a[:,num]
df = df.drop(columns = 'source')


new_sourceclass = np.asarray(df['source_class']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_sourceclass)
for num,i in enumerate(np.unique(df['source_class'])):
    df['sourceclass_'+str(i)] = a[:,num]
df = df.drop(columns = 'source_class')


new_wpt = np.asarray(df['waterpoint_type']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_wpt)
for num,i in enumerate(np.unique(df['waterpoint_type'])):
    df['wpttype_'+str(i)] = a[:,num]
df = df.drop(columns = 'waterpoint_type')

## predicting functionality
df_functional = df.copy()
new_qual = np.asarray(df_functional['quality_group']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_qual)
for num,i in enumerate(np.unique(df_functional['quality_group'])):
    df_functional['quality_'+str(i)] = a[:,num]
df_functional = df_functional.drop(columns = 'quality_group')


new_quant = np.asarray(df_functional['quantity_group']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_quant)
for num,i in enumerate(np.unique(df_functional['quantity_group'])):
    df_functional['quantity_'+str(i)] = a[:,num]
df_functional = df_functional.drop(columns = 'quantity_group')


###### Functionality Prediction #######

#filling NANs with mean of variable
df_func = df_functional.fillna(df_functional.mean())

#separating into test (25%) and train (75%)
x_train, x_test, y_train, y_test = train_test_split(df_func.drop(columns='labels'), df_func['labels'], test_size=0.25, random_state=0)

#three classes of our output variable
classes=['Functional', 'Needs Repair', 'Non-Functional']

#Encoding the three output classes into integers

enc = preprocessing.LabelEncoder()
enc.fit(y_test)
test_labels = enc.transform(y_test)

# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

## LR
logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

lr_acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(lr_acc, classes,title = 'Logistic Regression', normalize=True)

#optimizing hyperparameters
"""
We used GridSearchCV which performs k-fold cross validation (k=5 for us) and searches
a grid of specified parameters to find the best parameters.
"""
param_dist = {
'penalty': ['l1','l2'],
'C': [0.001,0.01,0.5,1]
}

lr_search= GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto'),param_dist,cv=5)
lr_search.fit(x_res,y_res)

lr_best = lr_search.best_estimator_ #best classifier found with GridSearchCV

lr_preds = lr_best.predict(x_test)
train_preds = lr_best.predict(x_res)

a = metrics.confusion_matrix(y_test,lr_preds)
plot_confusion_matrix(a, classes,title = 'Logistic Regression with L2 Penalty, C = 1'), normalize=True)

#calculating microaveraged F1 scores for train and test
f1_LR_train = metrics.f1_score(y_res,train_preds,average = 'micro')
f1_LR_test = metrics.f1_score(y_test,lr_preds,average = 'micro')

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

#getting the indices for the 5-fold cross validation of the test set (25% of original data)
kf = KFold(n_splits=5)
k_indices = []
for _, test_index in kf.split(x_test):
    k_indices.append(test_index)

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = lr_best.predict(x_res)
    test_preds = lr_best.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### RF optimizing hyperparameters
parameters = {'n_estimators':(100,70,50),'max_depth':(30,25,20,5)}
rf = RandomForestClassifier(criterion = 'entropy',min_samples_leaf = 5, min_samples_split = 10)
rf_cv = GridSearchCV(rf,parameters,cv=5,verbose=3)
rf_cv.fit(x_res,y_res)

rf_best = rf_cv.best_estimator_

#train and test predictions
train_rf_pred = rf_best.predict(x_res)
rf_pred = rf_best.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_rf_train = metrics.f1_score(y_res,train_rf_pred,average = 'micro')
f1_rf_test = metrics.f1_score(y_test,rf_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(a, classes,title = 'Random Forest, 70 Estimators', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = rf_best.predict(x_res)
    test_preds = rf_best.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Bagging optimizing hyperparameters
bag = BaggingClassifier()
bag.fit(x_res,y_res)
bag_pred = bag.predict(x_test)
acc = metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(acc, classes,
                      title='Bagging', normalize=True)

parameters = {'n_estimators':(100,70,50),'max_samples':(20,10,5),'max_features':(138,70,25)}
bag_cv = GridSearchCV(bag,parameters,cv=5,verbose=3)
bag_cv.fit(x_res,y_res)

bag_best = bag_cv.best_estimator_

#train and test predictions
train_bag_pred = bag_best.predict(x_res)
bag_pred = bag_best.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_bag_train = metrics.f1_score(y_res,train_bag_pred,average = 'micro')
f1_bag_test = metrics.f1_score(y_test,bag_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(a, classes,title = 'Bagging', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = bag_best.predict(x_res)
    test_preds = bag_best.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Adaboost
ab = AdaBoostClassifier(n_estimators=35)
ab.fit(x_res,y_res)
ab_pred = ab.predict(x_test)
acc = metrics.confusion_matrix(y_test,ab_pred)
plot_confusion_matrix(acc, classes,
                      title='AdaBoost', normalize=True)


## Neural net optimizing hyperparameters

nnet = MLPClassifier(alpha=1e-5)

parameters ={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(138,60,2), (100,10), (60,5,1), (75,30,5), (138)],
'activation': ["logistic", "tanh"]
}

nn_cv = GridSearchCV(nnet,parameters,cv=5,verbose=3)
nn_cv.fit(x_res,y_res)

nn_best = nn_cv.best_estimator_

#train and test predictions
train_nn_pred = nn_best.predict(x_res)
nn_pred = nn_best.predict(x_test)

a =metrics.confusion_matrix(y_test,nn_pred)
plot_confusion_matrix(a, classes,title = 'Neural Network', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = nn_best.predict(x_res)
    test_preds = nn_best.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### SVM hyperparameters
svm = SVC(gamma='scale')

parameters ={
'C': (0.01,0.1,1),
'kernel': ("rbf", "sigmoid","poly")
}

svm_cv = GridSearchCV(svm,parameters,cv=5,verbose=3)
svm_cv.fit(x_res,y_res)

svm_best = svm_cv.best_estimator_

#train and test predictions
train_svm_pred = svm_best.predict(x_res)
svm_pred = svm_best.predict(x_test)

a =metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(a, classes,title = 'SVM', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = svm_best.predict(x_res)
    test_preds = svm_best.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### combining LR, NN, RF with Ensemble - Voting Classifier

vote_func = VotingClassifier(estimators=[('lr', lr_best), ('rf', rf_best), ('nn', nn_best)])
vote_func.fit(x_res,y_res)


vote_preds_f = vote_func.predict(x_test) #test predictions
vote_preds_t = vote_func.predict(x_res) #train predictions

a = metrics.confusion_matrix(y_test,vote_preds_f)
plot_confusion_matrix(a, classes,title = 'Voting Classifier', normalize=True)


############# Water Quantity #############
df_quant = df.copy()

#OHE process for water quality and functionality - used as predictors for water
# quantity
new_qual = np.asarray(df_quant['quality_group']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_qual)
for num,i in enumerate(np.unique(df_quant['quality_group'])):
    df_quant['quality_'+str(i)] = a[:,num]
df_quant = df_quant.drop(columns = 'quality_group')


new_func = np.asarray(df_quant['labels']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False,categories='auto')
a = enc.fit_transform(new_func)
for num,i in enumerate(np.unique(df_quant['labels'])):
    df_quant[str(i)] = a[:,num]
df_quant = df_quant.drop(columns = 'labels')

#filling NANs with mean of variable
df_quant = df_quant.fillna(df_quant.mean())

#separating into test (25%) and train (75%)
x_train, x_test, y_train, y_test = train_test_split(df_quant.drop(columns='quantity_group'), df_quant['quantity_group'], test_size=0.25)

#five classes of our output variable
classes_quant = ['Dry', 'Enough', 'Insufficient', 'Seasonal','Unknown']

#Encoding the five output classes into integers
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
train_labels = enc.transform(y_train)

# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

## LR
logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

lr_acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(lr_acc, classes_quant,title = 'Logistic Regression', normalize=True)

#optimizing hyperparameters
"""
We used GridSearchCV which performs k-fold cross validation (k=5 for us) and searches
a grid of specified parameters to find the best parameters.
"""
param_dist = {
'penalty': ['l1','l2'],
'C': [0.001,0.01,0.5,1]
}

lr_search= GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto'),param_dist,cv=5)
lr_search.fit(x_res,y_res)

lr_best_qt = lr_search.best_estimator_ #best classifier found with GridSearchCV

lr_preds = lr_best_qt.predict(x_test)
train_preds = lr_best_qt.predict(x_res)

a = metrics.confusion_matrix(y_test,lr_preds)
plot_confusion_matrix(a, classes_quant,title = 'Logistic Regression with L2 Penalty, C = 1'), normalize=True)

#calculating microaveraged F1 scores for train and test
f1_LR_train = metrics.f1_score(y_res,train_preds,average = 'micro')
f1_LR_test = metrics.f1_score(y_test,lr_preds,average = 'micro')

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

#getting the indices for the 5-fold cross validation of the test set (25% of original data)
kf = KFold(n_splits=5)
k_indices = []
for _, test_index in kf.split(x_test):
    k_indices.append(test_index)

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = lr_best_qt.predict(x_res)
    test_preds = lr_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### RF optimizing hyperparameters
parameters = {'n_estimators':(100,75,45,20),'max_depth':(45,30,25,20)}
rf = RandomForestClassifier(criterion = 'entropy',min_samples_leaf = 5, min_samples_split = 10)
rf_cv = GridSearchCV(rf,parameters,cv=5,verbose=3)
rf_cv.fit(x_res,y_res)

rf_best_qt = rf_cv.best_estimator_

#train and test predictions
train_rf_pred = rf_best_qt.predict(x_res)
rf_pred = rf_best_qt.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_rf_train = metrics.f1_score(y_res,train_rf_pred,average = 'micro')
f1_rf_test = metrics.f1_score(y_test,rf_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(a, classes_quant,title = 'Random Forest, 45 Estimators', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = rf_best_qt.predict(x_res)
    test_preds = rf_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Bagging optimizing hyperparameters
bag = BaggingClassifier()
bag.fit(x_res,y_res)
bag_pred = bag.predict(x_test)
acc = metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='Bagging', normalize=True)

parameters = {'n_estimators':(100,75,45,20),'max_samples':(45,20,10,5),'max_features':(138,70,25)}
bag_cv = GridSearchCV(bag,parameters,cv=5,verbose=3)
bag_cv.fit(x_res,y_res)

bag_best_qt = bag_cv.best_estimator_

#train and test predictions
train_bag_pred = bag_best_qt.predict(x_res)
bag_pred = bag_best_qt.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_bag_train = metrics.f1_score(y_res,train_bag_pred,average = 'micro')
f1_bag_test = metrics.f1_score(y_test,bag_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(a, classes_quant,title = 'Bagging', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = bag_best_qt.predict(x_res)
    test_preds = bag_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Adaboost
ab = AdaBoostClassifier(n_estimators=35)
ab.fit(x_res,y_res)
ab_pred = ab.predict(x_test)
acc = metrics.confusion_matrix(y_test,ab_pred)
plot_confusion_matrix(acc, classes,
                      title='AdaBoost', normalize=True)


## Neural net optimizing hyperparameters

nnet = MLPClassifier(alpha=1e-5)

parameters ={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(137,50,2), (80,10), (60,15), (75,30,5), (137)],
'activation': ["logistic", "tanh"]
}

nn_cv = GridSearchCV(nnet,parameters,cv=5,verbose=3)
nn_cv.fit(x_res,y_res)

nn_best_qt = nn_cv.best_estimator_

#train and test predictions
train_nn_pred = nn_best_qt.predict(x_res)
nn_pred = nn_best_qt.predict(x_test)

a =metrics.confusion_matrix(y_test,nn_pred)
plot_confusion_matrix(a, classes,title = 'Neural Network', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = nn_best_qt.predict(x_res)
    test_preds = nn_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### SVM hyperparameters
svm = SVC(gamma='scale')

parameters ={
'C': (0.01,0.1,1),
'kernel': ("rbf", "sigmoid","poly")
}

svm_cv = GridSearchCV(svm,parameters,cv=5,verbose=3)
svm_cv.fit(x_res,y_res)

svm_best_qt = svm_cv.best_estimator_

#train and test predictions
train_svm_pred = svm_best_qt.predict(x_res)
svm_pred = svm_best_qt.predict(x_test)

a =metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(a, classes,title = 'SVM', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = svm_best_qt.predict(x_res)
    test_preds = svm_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### combining LR, NN, RF with Ensemble - Voting Classifier

vote_quant = VotingClassifier(estimators=[('lr', lr_best_qt), ('rf', rf_best_qt), ('nn', nn_best_qt)])
vote_quant.fit(x_res,y_res)


vote_preds_f = vote_quant.predict(x_test) #test predictions
vote_preds_t = vote_quant.predict(x_res) #train predictions

a = metrics.confusion_matrix(y_test,vote_preds_f)
plot_confusion_matrix(a, classes_qual,title = 'Voting Classifier', normalize=True)


############# Water Quality #############
df_qual = df.copy()

#OHE process for quantity and functionality - used as predictors for water quality
new_quant = np.asarray(df_qual['quantity_group']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False)
a = enc.fit_transform(new_quant)
for num,i in enumerate(np.unique(df_qual['quantity_group'])):
    df_qual['quantity_'+str(i)] = a[:,num]
df_qual = df_qual.drop(columns = 'quantity_group')


new_func = np.asarray(df_qual['labels']).reshape(-1,1)
enc = preprocessing.OneHotEncoder(sparse= False)
a = enc.fit_transform(new_func)
for num,i in enumerate(np.unique(df_qual['labels'])):
    df_qual[str(i)] = a[:,num]
df_qual = df_qual.drop(columns = 'labels')

#filling NANs with mean of variable
df_qual = df_qual.fillna(df_qual.mean())

#separating into test (25%) and train (75%)
x_train, x_test, y_train, y_test = train_test_split(df_qual.drop(columns='quality_group'), df_qual['quality_group'], test_size=0.25)

#six classes of our output variable
classes_qual = ['Colored', 'Fluoride', 'Good', 'Milky','Salty', 'Unknown']

#Encoding the output classes into integers
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
train_labels = enc.transform(y_train)

# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

## LR
logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

lr_acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(lr_acc, classes,title = 'Logistic Regression', normalize=True)

#optimizing hyperparameters
"""
We used GridSearchCV which performs k-fold cross validation (k=5 for us) and searches
a grid of specified parameters to find the best parameters.
"""
param_dist = {
'penalty': ['l1','l2'],
'C': [0.001,0.01,0.5,1]
}

lr_search= GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto'),param_dist,cv=5)
lr_search.fit(x_res,y_res)

lr_best_qt = lr_search.best_estimator_ #best classifier found with GridSearchCV

lr_preds = lr_best_qt.predict(x_test)
train_preds = lr_best_qt.predict(x_res)

a = metrics.confusion_matrix(y_test,lr_preds)
plot_confusion_matrix(a, classes,title = 'Logistic Regression with L2 Penalty, C = 1'), normalize=True)

#calculating microaveraged F1 scores for train and test
f1_LR_train = metrics.f1_score(y_res,train_preds,average = 'micro')
f1_LR_test = metrics.f1_score(y_test,lr_preds,average = 'micro')

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

#getting the indices for the 5-fold cross validation of the test set (25% of original data)
kf = KFold(n_splits=5)
k_indices = []
for _, test_index in kf.split(x_test):
    k_indices.append(test_index)

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = lr_best_qt.predict(x_res)
    test_preds = lr_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### RF optimizing hyperparameters
parameters = {'n_estimators':(100,75,45,20),'max_depth':(45,30,25,20)}
rf = RandomForestClassifier(criterion = 'entropy',min_samples_leaf = 5, min_samples_split = 10)
rf_cv = GridSearchCV(rf,parameters,cv=5,verbose=3)
rf_cv.fit(x_res,y_res)

rf_best_qt = rf_cv.best_estimator_

#train and test predictions
train_rf_pred = rf_best_qt.predict(x_res)
rf_pred = rf_best_qt.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_rf_train = metrics.f1_score(y_res,train_rf_pred,average = 'micro')
f1_rf_test = metrics.f1_score(y_test,rf_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(a, classes_qual,title = 'Random Forest, 45 Estimators', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = rf_best_qt.predict(x_res)
    test_preds = rf_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Bagging optimizing hyperparameters
bag = BaggingClassifier()
bag.fit(x_res,y_res)
bag_pred = bag.predict(x_test)
acc = metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(acc, classes,
                      title='Bagging', normalize=True)

parameters = {'n_estimators':(100,75,45,20),'max_samples':(45,20,10,5),'max_features':(138,70,25)}
bag_cv = GridSearchCV(bag,parameters,cv=5,verbose=3)
bag_cv.fit(x_res,y_res)

bag_best_qt = bag_cv.best_estimator_

#train and test predictions
train_bag_pred = bag_best_qt.predict(x_res)
bag_pred = bag_best_qt.predict(x_test)

#calculating microaveraged F1 scores for train and test
f1_bag_train = metrics.f1_score(y_res,train_bag_pred,average = 'micro')
f1_bag_test = metrics.f1_score(y_test,bag_pred,average = 'micro')

a =metrics.confusion_matrix(y_test,bag_pred)
plot_confusion_matrix(a, classes,title = 'Bagging', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = bag_best_qt.predict(x_res)
    test_preds = bag_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### Adaboost
ab = AdaBoostClassifier(n_estimators=35)
ab.fit(x_res,y_res)
ab_pred = ab.predict(x_test)
acc = metrics.confusion_matrix(y_test,ab_pred)
plot_confusion_matrix(acc, classes,
                      title='AdaBoost', normalize=True)


## Neural net optimizing hyperparameters

nnet = MLPClassifier(alpha=1e-5)

parameters ={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(137,50,2), (80,10), (60,15), (75,30,5), (137)],
'activation': ["logistic", "tanh"]
}

nn_cv = GridSearchCV(nnet,parameters,cv=5,verbose=3)
nn_cv.fit(x_res,y_res)

nn_best_qt = nn_cv.best_estimator_

#train and test predictions
train_nn_pred = nn_best_qt.predict(x_res)
nn_pred = nn_best_qt.predict(x_test)

a =metrics.confusion_matrix(y_test,nn_pred)
plot_confusion_matrix(a, classes,title = 'Neural Network', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = nn_best_qt.predict(x_res)
    test_preds = nn_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### SVM hyperparameters
svm = SVC(gamma='scale')

parameters ={
'C': (0.01,0.1,1),
'kernel': ("rbf", "sigmoid","poly")
}

svm_cv = GridSearchCV(svm,parameters,cv=5,verbose=3)
svm_cv.fit(x_res,y_res)

svm_best_qt = svm_cv.best_estimator_

#train and test predictions
train_svm_pred = svm_best_qt.predict(x_res)
svm_pred = svm_best_qt.predict(x_test)

a =metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(a, classes,title = 'SVM', normalize=True)

#Performing 5-fold CV on test set:
f1_train = []
f1_test = []
mcc_train = []
mcc_test = []

for i in range(5):
    inds = k_indices[i]
    x_t,y_t = np.asarray(x_test)[inds],np.asarray(y_test)[inds]
    train_preds = svm_best_qt.predict(x_res)
    test_preds = svm_best_qt.predict(x_t)
    f1_train.append(metrics.f1_score(y_res,train_preds,average = "micro"))
    f1_test.append(metrics.f1_score(y_t,test_preds,average = "micro"))
    mcc_train.append(metrics.matthews_corrcoef(y_res,train_preds)
    mcc_test.append(metrics.matthews_corrcoef(y_t,test_preds)
    print(i)

print(np.nanmean(f1_test))
print(np.nanmean(mcc_test))

### combining LR, NN, RF with Ensemble - Voting Classifier

vote_quant = VotingClassifier(estimators=[('lr', lr_best_qt), ('rf', rf_best_qt), ('nn', nn_best_qt)])
vote_quant.fit(x_res,y_res)


vote_preds_f = vote_quant.predict(x_test) #test predictions
vote_preds_t = vote_quant.predict(x_res) #train predictions

a = metrics.confusion_matrix(y_test,vote_preds_f)
plot_confusion_matrix(a, classes_qual,title = 'Voting Classifier', normalize=True)
