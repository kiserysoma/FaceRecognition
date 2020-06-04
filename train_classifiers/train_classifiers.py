import os
import pandas as pd
import face_recognition
import cv2
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

GET_EMBEDDINGS = True
IMAGE_PATH = '/media/Projects/shared/face_dataset_v2/'

images = sorted(os.listdir(IMAGE_PATH))
print(len(images))
IDs = []
indexes = []
buffer = []

if GET_EMBEDDINGS: 
    embeddings = []
    imgs = []
    i_ = 0
    for i in images:
        print(i_)
        i_ += 1
        img = cv2.imread(IMAGE_PATH + i)
        embedding = face_recognition.api.face_encodings(img)
        if len(embedding) == 1:
            embeddings.append(embedding)
            imgs.append(i)
    pickle.dump(embeddings, open('face_dataset_embeddings_v2.p', 'wb') )
    pickle.dump(imgs, open('img_names_v2.p', 'wb') )

imgs = pickle.load( open ('img_names_v2.p', 'rb') )
embeddings = pickle.load( open ('face_dataset_embeddings_v2.p', 'rb') )

for i in range(len(imgs)):
    index = imgs[i][0:5].find('_')
    ID = imgs[i][0:index]
    if ID not in IDs:
        indexes.append(i)
        IDs.append(ID)
    buffer.append(ID)
most_common = [item for item in Counter(buffer).most_common()]
#print(most_common)

if GET_EMBEDDINGS:
    dfs = []
    for i in range(len(embeddings)):
        d = pd.DataFrame(data=embeddings[i])
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle("face_dataset_embeddings_df_v2.p")

df = pd.read_pickle("face_dataset_embeddings_df_v2.p")
print(len(IDs))
print(most_common)
print("Used images:",len(imgs))

for i in range(len(IDs)):

    #Create labels
    labels = []
    if i == len(IDs)-1:
        for j in range(len(df.index.values)):
            if j >= indexes[i]:
                labels.append(1)
            else:
                labels.append(0)
    else:
        for j in range(len(df.index.values)):
            if j >= indexes[i] and j < indexes[i+1]:
                labels.append(1)
            else:
                labels.append(0)

    #Train RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3)
    rf_classifier = RandomForestClassifier(n_estimators=400)
    rf_classifier.fit(X_train,y_train)
    y_pred = rf_classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix: \n", metrics.confusion_matrix(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

    filename = '/media/Projects/shared/Classifiers/' + str(IDs[i]) + '_RFclassifier.p'
    pickle.dump(imgs, open(filename, 'wb') )