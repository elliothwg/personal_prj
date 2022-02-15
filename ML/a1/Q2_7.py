import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings(action='ignore')

# read dataset from files
def read_csv(path):
    rawdata = pd.read_csv(path)

    return rawdata

path=["d:/1_breast_cancer.csv","d:/2_iris.csv","d:/3_relax.csv","d:/4_skin.csv"]

# repeat the process with each file
for path_value in path:
    rawdata=read_csv(path_value)
    predictors = rawdata.iloc[:, :-1]
    classes = rawdata.iloc[:, -1]

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
# print
    print("----",path_value,"----")
    accruacy_table=pd.DataFrame()
    accruacy_table.insert(loc=0, column='Neurons', value=neuron_list)
    accruacy_table.insert(loc=1, column='Accuracy', value=accuracy_list)
    print(accruacy_table)
