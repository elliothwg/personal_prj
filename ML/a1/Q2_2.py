import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings(action='ignore')

# read dataset from file
path="d:/pima-indians-diabetes.csv"
rawdata=pd.read_csv(path,skiprows=9,names=['pregnant','glucose','blood pressure','triceps','insulin','body mass','diabetes','age','class'])
nrow,ncol=rawdata.shape
predictors=rawdata.iloc[:,:-1]
classes=rawdata.iloc[:,-1]

# divide the result with training and testing
pred_train, pred_test, tar_train, tar_test  =train_test_split(predictors,classes,test_size=0.3,random_state=0,stratify=classes)

# MLP
MLP_clf = MLPClassifier(hidden_layer_sizes=(20),max_iter=150)
MLP_clf = MLP_clf.fit(pred_train,np.ravel(tar_train,order='C'))

# preparation for epoch
N_TRAIN_SAMPLES = pred_train.shape[0]
N_EPOCHS = 150
N_BATCH = 400
N_CLASSES = np.unique(tar_train)
scores_train = []
scores_test = []
mlploss = []

# EPOCH
epoch = 0

while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    # # SHUFFLING
    random_perm = np.random.randint(0,N_TRAIN_SAMPLES-N_BATCH)
    MLP_clf.partial_fit(pred_train.iloc[random_perm:random_perm+N_BATCH], tar_train.iloc[random_perm:random_perm+N_BATCH],classes=N_CLASSES)
    scores_train.append(1 - MLP_clf.score(pred_train, tar_train))

    # SCORE TEST
    scores_test.append(1 - MLP_clf.score(pred_test, tar_test))

    # compute loss
    mlploss.append(MLP_clf.loss_)
    epoch += 1

""" Plot """
fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(scores_train)
ax[0].set_title('Train Error')
ax[1].plot(mlploss)
ax[1].set_title('Train Loss')
ax[2].plot(scores_test)
ax[2].set_title('Test Error')
fig.suptitle("Error vs Loss over epochs", fontsize=14)
plt.show()
