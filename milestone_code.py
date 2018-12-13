#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 06:44:36 2018

@author: jackieff, vkozlow, margalan

CS 229 Final Project code
All instructions included as comments within the code
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

#filling NANs with mean of variable
df_func = df_functional.fillna(df_functional.mean())

#separating into test (25%) and train (75%)
x_train, x_test, y_train, y_test = train_test_split(df_func.drop(columns='labels'), df_func['labels'], test_size=0.25, random_state=0)

#three classes of our output variable
classes=['Functional', 'Needs Repair', 'Non-Functional']

#Encoding the three output classes into integers
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
train_labels = enc.transform(y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_test)
test_labels = enc.transform(y_test)

#Logistic Regression for Functionality Prediction
logreg = LogisticRegression()
logreg.fit(x_train,train_labels)
y_pred=logreg.predict(x_test)

acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(acc, classes,
                      title = 'Logistic Regression Confusion Matrix', normalize=True)

probs = logreg.predict_proba(x_test)

#examining predicted probabilities for each class
plt.figure(figsize = (10,6))
plt.bar(range(14850),probs[:,0], color = 'green', label = 'Functional')
plt.bar(range(14850),probs[:,2], color = 'red', label = 'Non Functional')
plt.bar(range(14850),probs[:,1], color = 'blue', label = 'Needs Repair')
plt.grid(True, which='both')
plt.legend()

"""
We wanted to inspect the probabilities to see if there were cases where the highest
and second highest probabilities were extremely close, in which case we know we
are not that confident that a given sample belongs in the highest probability class.
"""


# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

lr_acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(lr_acc, classes,title = 'Logistic Regression - SMOTE Confusion Matrix', normalize=True)


#### GDA #####
clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
gda_pred = clf.predict(x_test)


acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes,
                      title='GDA Confusion Matrix', normalize=True)

# GDA with SMOTE
clf = LinearDiscriminantAnalysis()
clf.fit(x_res, y_res)
gda_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes,
                      title='GDA - SMOTE Confusion Matrix', normalize=True)


### SVM with original data
clf = SVC(gamma='scale')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(acc, classes,
                     title='SVM RBF Confusion Matrix', normalize=True)

### SVM with SMOTE
clf2 = SVC(gamma='scale')
clf2.fit(x_res, y_res)
svm_pred2 = clf2.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred2)
plot_confusion_matrix(acc, classes,
                     title='SVM RBF - SMOTE Confusion Matrix', normalize=True)


### RF with original data
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes,
                      title='Random Forest Confusion Matrix', normalize=True)


### RF with SMOTE
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_res,y_res)
rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes,
                      title='Random Forest - SMOTE Confusion Matrix', normalize=True)


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

#Encoding the three output classes into integers
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
train_labels = enc.transform(y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_test)
test_labels = enc.transform(y_test)


### LogReg with original data
logreg = LogisticRegression()
logreg.fit(x_train,train_labels)
y_pred=logreg.predict(x_test)

acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(acc, classes,
                      title = 'Logistic Regression Confusion Matrix', normalize=True)


# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(acc, classes_quant,title = 'Logistic Regression - SMOTE Confusion Matrix', normalize=True)


#### GDA with original data
clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
gda_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes_quant,
                      title='GDA Confusion Matrix', normalize=True)

### GDA with SMOTE
clf = LinearDiscriminantAnalysis()
clf.fit(x_res, y_res)
gda_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes_quant,
                      title='GDA - SMOTE Confusion Matrix', normalize=True)


### SVM with original data
clf = SVC(gamma='scale')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='SVM RBF Confusion Matrix', normalize=True)

### SVM with SMOTE
clf2 = SVC(gamma='scale')
clf2.fit(x_res, y_res)
svm_pred2 = clf2.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred2)
plot_confusion_matrix(acc, classes_quant,
                      title='SVM RBF - SMOTE Confusion Matrix', normalize=True)


### RF with original data
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='Random Forest Confusion Matrix', normalize=True)


### RF with SMOTE
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_res,y_res)

rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='Random Forest - SMOTE Confusion Matrix', normalize=True)

############# Water Quality Prediction #############
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

enc = preprocessing.LabelEncoder()
enc.fit(y_test)
test_labels = enc.transform(y_test)

# Normal logreg
### LogReg with original data
logreg = LogisticRegression()
logreg.fit(x_train,train_labels)
y_pred=logreg.predict(x_test)

acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(acc, classes,
                      title = 'Logistic Regression Confusion Matrix', normalize=True)

# SMOTE resampling to deal with class imbalance
sm = SMOTE()
x_res,y_res = sm.fit_resample(x_train,y_train)

enc = preprocessing.LabelEncoder()
enc.fit(y_res)
train_labels = enc.transform(y_res)

logreg = LogisticRegression()
logreg.fit(x_res,train_labels)
y_pred=logreg.predict(x_test)

acc = metrics.confusion_matrix(test_labels,y_pred)

plot_confusion_matrix(acc, classes_quant,title = 'Logistic Regression - SMOTE Confusion Matrix', normalize=True)


#### GDA with original data
clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
gda_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes_quant,
                      title='GDA Confusion Matrix', normalize=True)

### GDA with SMOTE
clf = LinearDiscriminantAnalysis()
clf.fit(x_res, y_res)
gda_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,gda_pred)

plot_confusion_matrix(acc, classes_quant,
                      title='GDA - SMOTE Confusion Matrix', normalize=True)


### SVM with original data
clf = SVC(gamma='scale')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='SVM RBF Confusion Matrix', normalize=True)

### SVM with SMOTE
clf2 = SVC(gamma='scale')
clf2.fit(x_res, y_res)
svm_pred2 = clf2.predict(x_test)

acc = metrics.confusion_matrix(y_test,svm_pred2)
plot_confusion_matrix(acc, classes_quant,
                      title='SVM RBF - SMOTE Confusion Matrix', normalize=True)


### RF with original data
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='Random Forest Confusion Matrix', normalize=True)


### RF with SMOTE
rf = RandomForestClassifier(n_estimators = 35, criterion = 'entropy', max_depth=35)
rf.fit(x_res,y_res)

rf_pred = rf.predict(x_test)
acc = metrics.confusion_matrix(y_test,rf_pred)
plot_confusion_matrix(acc, classes_quant,
                      title='Random Forest - SMOTE Confusion Matrix', normalize=True)


##### MAPS #####
"""
The three blocks of code below will plot each of our output variables onto
individual maps. All points are colored by their class, indicated in the legend
at the bottom of each map.  
"""
ullat = np.nanmax(df['latitude'])
ullon = np.nanmin(df['longitude'])
lrlat = np.nanmin(df['latitude'])
lrlon = np.nanmax(df['longitude'])

lat = np.arange(ullat, lrlat,0.05)
lon = np.arange(ullon,lrlon,0.05)
x,y=np.meshgrid(lon,lat)

longs = np.asarray(df['longitude'])
lats = np.asarray(df['latitude'])

# FUNCTIONALITY
plt.figure(figsize = (10,8))
m = Basemap(projection='cyl', llcrnrlat=lrlat , urcrnrlat=ullat , llcrnrlon= ullon, urcrnrlon=lrlon ,resolution = 'i', lon_0=0)
m.drawmapboundary(fill_color='skyblue')
    # fill continents, set lake color same as ocean color.
m.fillcontinents(color='tan',lake_color='skyblue')

plt.scatter(longs[df['labels']=='functional'],lats[df['labels']=='functional'], c = 'green', s =2, label = 'Functional', zorder=12, alpha = 0.3)
plt.scatter(longs[df['labels']=='functional needs repair'],lats[df['labels']=='functional needs repair'], c = 'purple', s =2, label = 'Needs Repair',zorder=12)
plt.scatter(longs[df['labels']=='non functional'],lats[df['labels']=='non functional'], c = 'red', s =2, label = 'Non-Functional', zorder=11)
#draw coastlines and map boundaries
m.drawcoastlines()
m.drawcountries()
plt.xticks(np.arange(np.ceil(ullon),lrlon,4))
plt.yticks(np.arange(-1,lrlat,-4))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
leg = plt.legend(markerscale = 5)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.title('Functionality of Water Pumps in Sub-Saharan Africa')
plt.tight_layout()

# QUALITY
plt.figure(figsize = (10,8))
m = Basemap(projection='cyl', llcrnrlat=lrlat , urcrnrlat=ullat , llcrnrlon= ullon, urcrnrlon=lrlon ,resolution = 'i', lon_0=0)
m.drawmapboundary(fill_color='skyblue')
    # fill continents, set lake color same as ocean color.
m.fillcontinents(color='tan',lake_color='skyblue')

color = ['teal', 'purple', 'green', 'white', 'yellow', 'red']
for num,val in enumerate(np.unique(df['quality_group'])):
    if val == 'good':
        plt.scatter(longs[df['quality_group']==val],lats[df['quality_group']==val], c = color[num], zorder = 2, s =2, label = val, alpha = 0.3)
    else:
        plt.scatter(longs[df['quality_group']==val],lats[df['quality_group']==val], c = color[num], zorder = num+10, s =2, label = val, alpha = 0.3)

#draw coastlines and map boundaries
m.drawcoastlines()
m.drawcountries()
plt.xticks(np.arange(np.ceil(ullon),lrlon,4))
plt.yticks(np.arange(-1,lrlat,-4))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
leg = plt.legend(markerscale = 5)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.title('Water Quality in Sub-Saharan Africa')
plt.tight_layout()

# QUANTITY
plt.figure(figsize = (10,8))
m = Basemap(projection='cyl', llcrnrlat=lrlat , urcrnrlat=ullat , llcrnrlon= ullon, urcrnrlon=lrlon ,resolution = 'i', lon_0=0)
m.drawmapboundary(fill_color='skyblue')
    # fill continents, set lake color same as ocean color.
m.fillcontinents(color='tan',lake_color='skyblue')

color = ['red', 'green', 'purple', 'orange', 'yellow']
for num,val in enumerate(np.unique(df['quantity_group'])):
    plt.scatter(longs[df['quantity_group']==val],lats[df['quantity_group']==val], c = color[num], zorder = num+10, s =3, label = val)

#draw coastlines and map boundaries
m.drawcoastlines()
m.drawcountries()
plt.xticks(np.arange(np.ceil(ullon),lrlon,4))
plt.yticks(np.arange(-1,lrlat,-4))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
leg = plt.legend(markerscale = 5)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.title('Water Quantity in Sub-Saharan Africa')
plt.tight_layout()
