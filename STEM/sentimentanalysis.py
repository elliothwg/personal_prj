import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


train_data = pd.read_json("d:/result/sentiment_analysis/training.json")
# train_data = pd.read_json("d:/result/sentiment_analysis/okt_result/1801-레드벨벳.json")




# train_data['label'].value_counts().plot(kind='bar')
# plt.show()
#
# test_data['label'].value_counts().plot(kind='bar')
# plt.show()
# print(train_data.groupby('label').size().reset_index(name='count'))
# print(test_data.groupby('label').size().reset_index(name='count'))


stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다','.','"','\n']

import konlpy
from konlpy.tag import Okt

okt = Okt()
X_train = []
# print(train_data)
for sentence in train_data[0]:
    # print(sentence)
    X_train.append(sentence)



# print(X_train[:3])
# print(X_test[:3])
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
max_words = 35000

tokenizer =Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)


# print(X_train[:3])

import numpy as np
y_train = []
y_test = []
for i in range(len(train_data[1])):
    if train_data[1].iloc[i] == 1:
        y_train.append([0, 0, 1])
    elif train_data[1].iloc[i] == 0:
        y_train.append([0, 1, 0])
    elif train_data[1].iloc[i] == -1:
        y_train.append([1, 0, 0])


y_train = np.array(y_train)


# print(y_train)

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
#
max_len = 20
# X_train = pad_sequences(X_train, maxlen=max_len)

# model = Sequential()
# model.add(Embedding(max_words, 100))
# model.add(LSTM(128))
# model.add(Dense(3, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

model=load_model('model.1801')

#
#
# test_files=["1801-장덕철.json","1801-멜로망스.json","1801-윤종신.json","1806-닐로.json","1806-방탄소년단.json","1808-숀.json","1808-방탄소년단.json","1808-TWICE.json"]
#
import os
path_dir="d:/result/sentiment_analysis/okt_result/"
test_files=os.listdir(path_dir)
result_file = open("d:/result/sentiment_analysis/final_result/summary.txt", 'w', -1, 'utf-8')
for test_file in test_files:
    test_data = pd.read_json("d:/result/sentiment_analysis/okt_result/"+test_file)
    # result_file = open("d:/result/sentiment_analysis/final_result/" + test_file.split('.')[0]+".txt", 'w', -1, 'utf-8')
    X_test = []
    y_test=[]

    for sentence in test_data[0]:
        X_test.append(sentence)

    X_test = tokenizer.texts_to_sequences(X_test)
    for i in range(len(test_data[1])):
        if test_data[1].iloc[i] == 1: y_test.append([0, 0, 1])
        elif test_data[1].iloc[i] == 0: y_test.append([0, 1, 0])
        elif test_data[1].iloc[i] == -1: y_test.append([1, 0, 0])
    y_test = np.array(y_test)
    X_test = pad_sequences(X_test, maxlen=max_len)
    print('\n'+test_file+' =>  accuracy rate : {:.2f}%'.format(model.evaluate(X_test,y_test)[1]*100))


    predict = model.predict(X_test)
    import numpy as np
    predict_labels = np.argmax(predict, axis=1)
    original_labels = np.argmax(y_test, axis=1)
    pos_ori=0
    neg_ori=0
    obj_ori=0
    pos_pre=0
    neg_pre=0
    obj_pre=0
    for i in range(0,len(original_labels)):
        print("Tweet : ", test_data[0].iloc[i], "/\t current : ", original_labels[i], "/\t predict : ", predict_labels[i])
        # result_file.write('\nTweet : '+str(test_data[0].iloc[i])+'/\t current : '+str(original_labels[i])+'/\t predict : '+str(predict_labels[i]))
        if original_labels[i]==2:
            pos_ori=pos_ori+1
        elif original_labels[i]==0:
            neg_ori = neg_ori + 1
        else:
            obj_ori=obj_ori+1
        if predict_labels[i] == 2:
            pos_pre = pos_pre + 1
        elif predict_labels[i] == 0:
            neg_pre = neg_pre + 1
        else:
            obj_pre = obj_pre + 1

    # result_file.write('\n\n' + test_file + str(' =>  accuracy rate : {:.2f}%'.format(model.evaluate(X_test, y_test)[1] * 100)))
    # result_file.write('\n\n<summary>')
    # result_file.write('\ntotal : '+str(pos_ori+obj_ori+neg_ori))
    # result_file.write('\noriginal poitive : ' + str(pos_ori))
    # result_file.write('\noriginal negative : ' + str(neg_ori))
    # result_file.write('\noriginal objectove : ' + str(obj_ori))
    # result_file.write('\npredict poitive : ' + str(pos_pre))
    # result_file.write('\npredict negative : ' + str(neg_pre))
    # result_file.write('\npredict objectove : ' + str(obj_pre))

    result_file.write('\n1808\t' + test_file + '\t'+str(pos_ori + obj_ori + neg_ori)+'\t'+str(pos_pre)+'\t'+str(neg_pre)+'\t'+str(obj_pre)+'\t'+str('{:.2f}%'.format(model.evaluate(X_test, y_test)[1] * 100)))
    # result_file.write('\n\n<summary>')
    # result_file.write('\ntotal : ' + str(pos_ori + obj_ori + neg_ori))
    # result_file.write('\noriginal poitive : ' + str(pos_ori))
    # result_file.write('\noriginal negative : ' + str(neg_ori))
    # result_file.write('\noriginal objectove : ' + str(obj_ori))
    # result_file.write('\npredict poitive : ' + str(pos_pre))
    # result_file.write('\npredict negative : ' + str(neg_pre))
    # result_file.write('\npredict objectove : ' + str(obj_pre))

result_file.close()