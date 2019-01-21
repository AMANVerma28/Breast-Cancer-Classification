import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import pickle
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV, cross_val_score # for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model

data = pd.read_csv("WBCD.csv",header=0)
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

y = data['diagnosis']
# Dropping columns that are highly correlated along with diagnosis column because it's label
drop_list = ['diagnosis', 'perimeter_mean','radius_mean','compactness_mean','concave points_mean', \
                'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',\
                'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
X = data.drop(drop_list,axis = 1 ) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=14)
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# Logistic Regression Model
# LRModel = LogisticRegression()
# LRModel.fit(X_train, y_train)
# pickle.dump(LRModel, open('LRModel.sav', 'wb'))
LRModel = pickle.load(open('LRModel.sav', 'rb'))
LRModel_accuracy = metrics.accuracy_score(LRModel.predict(X_test), y_test)
print("Accuracy of LR Model:", LRModel_accuracy)

# Decision Tree Classifier
# DTModel = DecisionTreeClassifier()
# DTModel.fit(X_train, y_train)
# pickle.dump(DTModel, open('DTModel.sav', 'wb'))
DTModel = pickle.load(open('DTModel.sav', 'rb'))
DTModel_accuracy = metrics.accuracy_score(DTModel.predict(X_test), y_test)
print("Accuracy of DT Model:", DTModel_accuracy)

# Random Forest Classifier
# RFModel = RandomForestClassifier()
# RFModel.fit(X_train, y_train)
# pickle.dump(RFModel, open('RFModel.sav', 'wb'))
RFModel = pickle.load(open('RFModel.sav', 'rb'))
RFModel_accuracy = metrics.accuracy_score(RFModel.predict(X_test), y_test)
print("Accuracy of RF Model:", RFModel_accuracy)

# KNN Classiifer
# KNNModel = KNeighborsClassifier()
# KNNModel.fit(X_train, y_train)
# pickle.dump(KNNModel, open('KNNModel.sav', 'wb'))
KNNModel = pickle.load(open('KNNModel.sav', 'rb'))
KNNModel_accuracy = metrics.accuracy_score(KNNModel.predict(X_test), y_test)
print("Accuracy of KNN Model:", KNNModel_accuracy)

# SVM Classifier
# SVMModel = svm.SVC(probability=True)
# SVMModel.fit(X_train, y_train)
# pickle.dump(SVMModel, open('SVMModel.sav', 'wb'))
SVMModel = pickle.load(open('SVMModel.sav', 'rb'))
SVMModel_accuracy = metrics.accuracy_score(SVMModel.predict(X_test), y_test)
print("Accuracy of SVM Model:", SVMModel_accuracy)

# MLP Classifier
# MLPModel = MLPClassifier()
# MLPModel.fit(X_train, y_train)
# pickle.dump(MLPModel, open('MLPModel.sav', 'wb'))
MLPModel = pickle.load(open('MLPModel.sav', 'rb'))
MLPModel_accuracy = metrics.accuracy_score(MLPModel.predict(X_test), y_test)
print("Accuracy of MLP Model:", MLPModel_accuracy)

# For plotting ROC curve
plt.figure(0).clf()
pred = LRModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="LR Model, auc="+str(auc))

pred = DTModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="DT Model, auc="+str(auc))

pred = RFModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="RF Model, auc="+str(auc))

pred = KNNModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="KNN Model, auc="+str(auc))

pred = SVMModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="SVM Model, auc="+str(auc))

pred = MLPModel.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr,tpr,label="MLP Model, auc="+str(auc))
plt.legend(loc=0)
plt.show()