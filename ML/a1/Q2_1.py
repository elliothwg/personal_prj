import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings(action='ignore')

# read the dataset from file.
path="d:/pima-indians-diabetes.csv"
rawdata=pd.read_csv(path,skiprows=9,names=['pregnant','glucose','blood pressure','triceps','insulin','body mass','diabetes','age','class'])
nrow,ncol=rawdata.shape
predictors=rawdata.iloc[:,:-1]
classes=rawdata.iloc[:,-1]

# divide the parts with training and testing
pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,classes,test_size=0.3,random_state=0,stratify=classes)

# MLP
MLP_clf = MLPClassifier(hidden_layer_sizes=(20),max_iter=150)
MLP_clf = MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))
MLP_predictions = MLP_clf.predict(pred_test)

# accuracy score of MLP
print("< Multi-Layer Perceptron >")
print("Accuracy score with MLP :", accuracy_score(tar_test, MLP_predictions))

# confusion matrix of MLP
print("  ----Confusion Matrix----")
MLP_cm=pd.DataFrame(confusion_matrix(tar_test,MLP_predictions))
print(MLP_cm)