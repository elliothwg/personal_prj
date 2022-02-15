import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings(action='ignore')


path="d:/Forest.xlsx"
rawdata=pd.read_excel(path)

# print("data summary")
# print(rawdata.describe())
nrow,ncol=rawdata.shape
# print(nrow,ncol)
# print("\n correlation Matrix")
# print(rawdata.corr())
class_le = LabelEncoder()
forest=class_le.fit_transform(rawdata['class'].values)
# print(forest)

# print(rawdata)
predictors=rawdata.iloc[:,1:]
# print("\n predictor")
# print(predictors)
# forest=rawdata.iloc[:,0]
# forest=le
# print("\n forest")
# print(forest)



pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,forest,test_size=0.3,random_state=0,stratify=forest)

# print(pred_train)
# print(pred_test)
# print(tar_train)
# print(tar_test)

dt_classifier = DecisionTreeClassifier(criterion="entropy",min_samples_split =50)#configure the classifier
dt_classifier = dt_classifier.fit(pred_train,tar_train)# train a decision tree model
dt_predictions = dt_classifier.predict(pred_test)


MLP_clf = MLPClassifier()
MLP_clf=MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))
MLP_predictions = MLP_clf.predict(pred_test)




dt_proba=dt_classifier.predict_proba(pred_test)
MLP_proba=MLP_clf.predict_proba(pred_test)


output_q2=pd.DataFrame()
output_q2.insert(loc=0,column='Decision Tree',value=np.ndarray.tolist(dt_proba[0]))
output_q2.insert(loc=1,column='MLP',value=np.ndarray.tolist(MLP_proba[0]))
output_q2.index=['Mixed','Hinoki','Other','Sugi']

print("--Probabilities of the first sample--")
print(output_q2)
