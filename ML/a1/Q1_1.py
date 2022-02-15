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


path="d:/Forest.xlsx"        # update it to the correct path
rawdata=pd.read_excel(path)
predictors=rawdata.iloc[:,1:]

class_le = LabelEncoder()
forest=class_le.fit_transform(rawdata['class'].values)   # replace the class values to numeric values


pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,forest,test_size=0.3,random_state=0,stratify=forest)


dt_classifier = DecisionTreeClassifier(criterion="entropy",min_samples_split =50) #configure the classifier
dt_classifier = dt_classifier.fit(pred_train, tar_train)   # train a decision tree model
dt_predictions = dt_classifier.predict(pred_test)

print("  < Decision Tree >")
print("Accuracy score with Decision Tree:", accuracy_score(tar_test, dt_predictions))
print("  ----Confusion Matrix----")
df_cm=pd.DataFrame(confusion_matrix(tar_test,dt_predictions,labels=[3,1,0,2]))    # re-order rows as 'Sugi','Hinoki','Mixed','Other'
df_cm.columns=['Sugi','Hinoki','Mixed','Other']
df_cm.index=['Sugi','Hinoki','Mixed','Other']
print(df_cm)



MLP_clf = MLPClassifier()
MLP_clf=MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))    # train
MLP_predictions = MLP_clf.predict(pred_test)
print("\n\n  < Multi-Layer Perceptron >")
print("Accuracy score with MLP :", accuracy_score(tar_test, MLP_predictions))
print("  ----Confusion Matrix----")
MLP_cm=pd.DataFrame(confusion_matrix(tar_test,MLP_predictions,labels=[3,1,0,2]))  # re-order rows as 'Sugi','Hinoki','Mixed','Other'
MLP_cm.columns=['Sugi','Hinoki','Mixed','Other']
MLP_cm.index=['Sugi','Hinoki','Mixed','Other']
print(MLP_cm)



dt_proba=dt_classifier.predict_proba(pred_test)      # probability of each sample and class (Decision Tree)
MLP_proba=MLP_clf.predict_proba(pred_test)          # probability of each sample and class (MLP)

print("\n\n--Probabilities of the first sample--")
output_q2=pd.DataFrame()
output_q2.insert(loc=0,column='Decision Tree',value=np.ndarray.tolist(dt_proba[0]))
output_q2.insert(loc=1,column='MLP',value=np.ndarray.tolist(MLP_proba[0]))
output_q2.index=['Mixed','Hinoki','Other','Sugi']
print(output_q2)