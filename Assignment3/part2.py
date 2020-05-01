import pandas as pd
import numpy as np
import pickle
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn import metrics

Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
	pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
	df = pd.read_csv(file)
	return df

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def trainOnAllData(df, clf):
	#Use this function for part 4, once you have selected the best model
	clf.fit(X1,Y1)
	print("TODO")

	saveBestModel(clf)
#def kfold_func():

df = readData("credit_train.csv")
#print(df)
X = []
y = []


for index, row in df.iterrows():
	#X.append(row)
	X.append(row[0:19])
	y.append(row[19])

arr = np.array(X)
arr2 = np.array(y)
#print(arr)
#print(arr2)

#cv = KFold(n_splits=10,random_state = None, shuffle =False)
#summ = 0
#for train_index, test_index in cv.split(arr):
	#print("Train Index: ", train_index, "\n")
	#print("Test Index: ", test_index)
	#X_train, X_test, y_train, y_test = arr[train_index], arr[test_index], arr2[train_index], arr2[test_index]
	#lg = LogisticRegression()
	#NB = GaussianNB()
	#SVM = svm.SVC(gamma='scale',probability = True)
	#DTC = tree.DecisionTreeClassifier()
	#RFC = RandomForestClassifier()
	
	#lg.fit(X_train,y_train)
	#NB.fit(X_train, y_train)
	#SVM.fit(X_train,y_train)
	#DTC.fit(X_train,y_train)
	#RFC.fit(X_train, y_train)
	#MLP.fit(X_train, y_train)
	#P.fit(X_train, y_train)
	#summ += roc_auc_score(y_test, RFC.predict_proba(X_test)[:,1])
	#print("Logistic Regression", roc_auc_score(y_test, lg.predict_proba(X_test)[:,1]))
	#print("Naive Bayes", roc_auc_score(y_test, NB.predict_proba(X_test)[:,1]))
	#print("Support Vector", roc_auc_score(y_test, SVM.predict_proba(X_test)[:,1]))
	#print("Decision Tree", roc_auc_score(y_test, DTC.predict_proba(X_test)[:,1]))
	#print("Random Forest Classification", roc_auc_score(y_test, RFC.predict_proba(X_test)[:,1]))
#print(summ/10)	

param_grid = {'n_estimators':[5, 10, 15, 20, 25, 50, 100, 500],'max_depth':[None,10,20,30,40,50,60,70,80,90,100,1000,10000]}

forest_reg = RandomForestClassifier()


grid_search = GridSearchCV(forest_reg,param_grid,cv=10)

grid_search.fit(X,y)

#print(grid_search.best_estimator_)

print("GridSearchCV for %d candidate parameter settings."
     % (len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



X1,Y1 = df[Features], df[Label]
#print(summ/10)
lg = LogisticRegression()
NB = GaussianNB()
SVM = svm.SVC()
DTC = tree.DecisionTreeClassifier()
RFC = RandomForestClassifier()
MLP = MLPClassifier()
ETC = ExtraTreesClassifier()
scores = cross_val_score(lg,X1,Y1,cv=10)
scores_2 = cross_val_score(NB,X1,Y1,cv=10)
scores_3 = cross_val_score(SVM,X1,Y1,cv=10)
scores_4 = cross_val_score(DTC,X1,Y1,cv=10)
scores_5 = cross_val_score(RFC,X1,Y1,cv=10)
scores_6 = cross_val_score(MLP,X1,Y1,cv=10)
scores_7 = cross_val_score(ETC,X1,Y1,cv=10)
print(scores)
print(scores.mean(),scores.std())
print(scores_2.mean(),scores_2.std())
print(scores_3.mean(),scores_3.std())
print(scores_4.mean(),scores_4.std())
print(scores_5.mean(),scores_5.std())
print(scores_6.mean(),scores_6.std())
print(scores_7.mean(),scores_7.std())

RF = RandomForestClassifier(n_estimators=50,max_depth=None)
RF.fit(X1,Y1)
#RF = LogisticRegression()
#RF.fit(X1,Y1)
Predicted = RF.predict(X1)
#y_pred = cross_val_predict(RF,X1,Y1,cv=10)
#y_pred = 
#print(y_pred)
#conf_mat = confusion_matrix(Y1,Predicted)
#print(conf_mat)
#print(metrics.classification_report(Y1,Predicted))



#param_grid = {'n_estimators':[5,10,20,25,30,40, 50, 100, 500],'max_depth':[None,10,100,1000]}

#forest_reg = RandomForestClassifier()



#grid_search = GridSearchCV(forest_reg,param_grid,cv=10)



#grid_search.fit(X1,Y1)

#print("GridSearchCV for %d candidate parameter settings."
     #% (len(grid_search.cv_results_['params'])))
#report(grid_search.cv_results_)

#param_grid_2 = {'C': [0.1,1,10,100,1000,10000,100000,1000000000], 'gamma':['auto','scale',0.01,0.10,0.5,1.0]}

#SVC = svm.SVC()

#grid_search_2 = GridSearchCV(SVC,param_grid= param_grid_2,cv = 10)

#grid_search_2.fit(X1,Y1)

#print("GridSearchCV for %d candidate parameter settings."
     #% (len(grid_search_2.cv_results_['params'])))
#report(grid_search_2.cv_results_)

df['Predicted'] = Predicted
df.to_csv(r'bestModel.output',index=False)
#print(df)
#f = RandomForestClassifier(n_estimators=100,max_depth=None)
#f.fit(X1,Y1)
#confusion_arr = f.predict_proba(X1)[:,1]
#print(confusion_matrix(Y1, f))

trainOnAllData(df,RF)
