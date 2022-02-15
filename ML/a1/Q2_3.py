import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings(action='ignore')

# read the dataset from file
path="d:/pima-indians-diabetes.csv"
rawdata=pd.read_csv(path,skiprows=9,names=['pregnant','glucose','blood pressure','triceps','insulin','body mass','diabetes','age','class'])
nrow,ncol=rawdata.shape
predictors=rawdata.iloc[:,:-1]
classes=rawdata.iloc[:,-1]

# divide the parts with training and testing
pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,classes,test_size=0.3,random_state=0,stratify=classes)

neuron_list=[]
accuracy_list=[]

# repeat the MLP with various numbers of hidden layers
for i in range(1,20):
    MLP_clf = MLPClassifier(hidden_layer_sizes=(20-i,i),max_iter=150)
    MLP_clf = MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))
    MLP_predictions = MLP_clf.predict(pred_test)
    neuron_list.append([20-i,i])
    accuracy_list.append(accuracy_score(tar_test, MLP_predictions))

# print the result
accuracy_table=pd.DataFrame()
accuracy_table.insert(loc=0, column='Neurons', value=neuron_list)
accuracy_table.insert(loc=1, column='Accuracy', value=accuracy_list)
print(accuracy_table)
