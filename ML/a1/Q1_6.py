import pandas as pd
import numpy as np
import collections
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings(action='ignore')

path="d:/Forest.xlsx"  # update it to the correct path
rawdata=pd.read_excel(path)
class_le = LabelEncoder()
forest=class_le.fit_transform(rawdata['class'].values)
predictors=rawdata.iloc[:,1:]

# divide the parts with training and testing
pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,forest,test_size=0.3,random_state=0,stratify=forest)

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion="entropy",min_samples_split =50)#configure the classifier
dt_classifier = dt_classifier.fit(pred_train, tar_train) # train a decision tree model
dt_predictions = dt_classifier.predict(pred_test)

# MLP
MLP_clf = MLPClassifier()
MLP_clf = MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))
MLP_predictions = MLP_clf.predict(pred_test)


## preparation for calculating total accuracy
# Generate the probability for each class, classifier and sample
dt_proba=dt_classifier.predict_proba(pred_test)
MLP_proba=MLP_clf.predict_proba(pred_test)

# count numbers of each class in total(test result array, DT array, MLP array) (ex [s:23,t:10,...])
total_class_cnt=collections.Counter(tar_test)
DT_total_class_cnt=collections.Counter(dt_predictions)
MLP_total_class_cnt=collections.Counter(MLP_predictions)

# if predicted class by each classfier and test result are same, save it to new compare_array. (ex. [1,0,1,1,1,0,....])
compare_DT_and_target=(dt_predictions==tar_test)
compare_MLP_and_target=(MLP_predictions==tar_test)

# generate blank arrays
cnt_DT_and_target= {0:0,1:0,2:0,3:0}
cnt_MLP_and_target= {0:0,1:0,2:0,3:0}

# generate blank arrays
prob_DT_and_target= {0:0,1:0,2:0,3:0}
prob_MLP_and_target= {0:0,1:0,2:0,3:0}

# count and save numbers (DT and MLP prediction == testing result) for calculating accuracy (ex. [s:23,t:11,...])
for i in range(0,len(tar_test)):
    if (compare_DT_and_target[i] == True):
        cnt_DT_and_target[int(tar_test[i])]=cnt_DT_and_target[int(tar_test[i])]+1
    if (compare_MLP_and_target[i] == True):
        cnt_MLP_and_target[int(tar_test[i])] = cnt_MLP_and_target[int(tar_test[i])] + 1

# save each class's probability of (DT and MLP)=(testing result) to prob_array
for k in total_class_cnt:
    prob_DT_and_target[k]=cnt_DT_and_target[k]/total_class_cnt[k]
    prob_MLP_and_target[k]=cnt_MLP_and_target[k]/total_class_cnt[k]


acc_cnt=0
result_class_list=np.array([])

for j in range(0,len(tar_test)):
# make the value for comparing : conditional probability  (ex 0.6*Pr(class=’s’|DT=’s’))
    dt_val=np.max(dt_proba[j])*prob_DT_and_target[int(np.argmax(dt_proba[j]))]
    MLP_val=np.max(MLP_proba[j])*prob_MLP_and_target[int(np.argmax(MLP_proba[j]))]
# if conditional probability of DT is bigger than MLP's, save the DT's class(index) to result_class_list
    if(dt_val>MLP_val):
        result_class_list= np.append(result_class_list,np.argmax(dt_proba[j]))
# if conditional probability of DT is smaller than MLP's, save the MLP's class(index) to result_class_list
    elif(MLP_val>dt_val):
        result_class_list= np.append(result_class_list,np.argmax(MLP_proba[j]))

# if the testing result and conditional probability are same,
# which means conditional probability's predicted value is correct,
# add 1 to 'acc_cnt' - for calculating total accuracy
    if(result_class_list[j]==tar_test[j]):
        acc_cnt=acc_cnt+1

# print and calculate the total accuracy (positive / total)
print("--Accuracy score : multiplying the maximum probability by its conditional probability--")
print(" => ",acc_cnt/len(tar_test))