import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings(action='ignore')

path="d:/Forest.xlsx"
rawdata=pd.read_excel(path)
class_le = LabelEncoder()
forest=class_le.fit_transform(rawdata['class'].values)
predictors=rawdata.iloc[:,1:]

pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,forest,test_size=0.3,random_state=0,stratify=forest)

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion="entropy",min_samples_split =50)
dt_classifier = dt_classifier.fit(pred_train, tar_train)

# MLP
MLP_clf = MLPClassifier()
MLP_clf=MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))


# Generate the probability for each class, classifier and sample
dt_proba=dt_classifier.predict_proba(pred_test)
MLP_proba=MLP_clf.predict_proba(pred_test)

# Generate the Average Aggregation values for each sample and class
avg_aggregate=(dt_proba+MLP_proba)/2

# Generate the empty array for calculating the Accuracy score
assign_class=np.ndarray((0,len(avg_aggregate)),int)

# Insert the class value which has the maximum probability of each sample
for i in range(0,len(avg_aggregate)):
    assign_class=np.append(assign_class,np.argmax(avg_aggregate[i]))

# Caculate the accuracy score by comparing them with actual class values
print("--Accuracy score with Average Aggregate of decision tree and MLP--")
accuracy_value=np.mean(tar_test==assign_class)
print(" => ",accuracy_value)

