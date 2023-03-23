import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import glob
import cv2
from keras.utils import to_categorical
import keras
from sklearn.model_selection import StratifiedShuffleSplit

train = pd.read_csv('../input/traina3/traina_3.csv')

train.head()

len(train)

train_list = [[Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png'),j] for i,j in zip(train.id_code[:5],train.diagnosis[:5])]
train_list

for i,j in train_list:
    plt.figure(figsize=(5,3))
    i = cv2.resize(np.asarray(i),(256,256))
    plt.title(j)
    plt.imshow(i)
    plt.show

x_train = [cv2.resize(np.asarray(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png')),(256,256)) for i in train.id_code]

x_train = np.array(x_train)

x_train = np.array(x_train)

y_train = train.diagnosis

y_train = to_categorical(y_train)
y_train

s = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=7)
s.get_n_splits(x_train, y_train)
for train_index, test_index in s.split(x_train, y_train):
    x_traino, x_testo = x_train[train_index], x_train[test_index]
    y_traino, y_testo = y_train[train_index], y_train[test_index]

m1 = keras.applications.densenet.DenseNet121(input_shape=(256,256,3),include_top=True,weights=None)

m1.summary()

m1.load_weights('../input/densenet-keras/DenseNet-BC-121-32.h5')

x = m1.layers[-2].output
d = keras.layers.Dense(512,activation='relu')(x)
e = keras.layers.Dense(5,activation='softmax')(d)

m2 = keras.models.Model(m1.input,e)

m2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

m2.fit(x_traino,y_traino,validation_data=(x_testo,y_testo),epochs=20)

test1 = []
test_rd = pd.read_csv('../input/testa3/testa_3.csv')

for i in test_rd.id_code:
    tp = np.array(cv2.resize(np.array(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png')),(256,256)))
    test1.append(tp)

test1 = np.array(test1)

np.random.seed(42)
res_1 = m2.predict(test1)

res_2 = []
for i in res_1:
    res_2.append(np.argmax(i))

test2 = pd.DataFrame({"id_code": test_rd["id_code"].values, "diagnosis": res_2})
test2.head(20)

test2.to_csv('submission.csv',index=False)

dtest = pd.read_csv('../input/traina-testa-compare-3/traina_testa_compare_3.csv')

A = test2["diagnosis"]

B = dtest["diagnosis"]

clu = ["Non Proliferative DR","Proliferative DR"]
v=0
v1=0
for i in A:
    if (i==0):
        v=v+1
    else:
        v1=v1+1
uo=[v,v1]
V = True
if V:
    sns.barplot(clu, uo, alpha=0.8, palette='magma')
    plt.xlabel("Severity of Diabetic Retinopathy")
    plt.ylabel("Count")
    plt.show()

C = []
for i in A:
    if(i==0):
        lu=0
        C.append(lu)
    else:
        lus=1
        C.append(lus)

D = []
for i in B:
    if(i==0):
        lu=0
        D.append(lu)
    else:
        lus=1
        D.append(lus)

from sklearn.metrics import accuracy_score

accuracy_score(C,D)

from sklearn.metrics import confusion_matrix

C_M = confusion_matrix(C,D)

import seaborn as sns

sns.heatmap(C_M, annot=True, fmt='.2f', cmap="Blues").set_title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

print (classification_report(C,D))

clu = ["Non-Proliferative","Mild", "Moderate", "Severe", "Proliferative" ]
ddr=0
ddr1=0
ddr2=0
ddr3=0
ddr4=0
for i in A:
    if (i==0):
        ddr=ddr+1
    elif(i==1):
        ddr1=ddr1+1
    elif(i==2):
        ddr2=ddr2+1
    elif(i==3):
        ddr3=ddr3+1
    elif(i==4):
        ddr4=ddr4+1    
uo=[ddr,ddr1,ddr2,ddr3,ddr4]
V = True
if V:
    
    plt.figure(figsize=(10,5))
    sns.barplot(clu, uo, alpha=0.8, palette='magma')
    plt.title('Severity')
    plt.ylabel('No of cases', fontsize=12)
    plt.xlabel('Types of Cases', fontsize=12)
    plt.show()

cf_matrix = confusion_matrix(A,B)

sns.heatmap(cf_matrix, annot=True, fmt='.2f', cmap="Blues").set_title('Confusion Matrix')
plt.show()

accuracy_score(A,B)

print (classification_report(A,B))
