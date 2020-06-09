import os
import pandas as pd
import face_recognition
import cv2
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

GET_EMBEDDINGS = False
IMAGE_PATH = '/media/Projects/shared/face_dataset_v2/'

images = sorted(os.listdir(IMAGE_PATH))
#print(len(images))
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
df2 = df.drop(columns=[50,86,37,110,58,81,13,83,55,89,48,49,69,20,108,98,71,90,82,117,24,19,126,32,76,41,34,16,10,14,42])
print('Modified embedding length:',len(df2.columns.values))

accuracy = []
accuracy_modified = []

recall = []
recall_modified = []

print(len(IDs),"different person")
#print(most_common)
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
    rf_classifier_original = RandomForestClassifier(n_estimators=400)
    rf_classifier_original.fit(X_train,y_train)
    y_pred = rf_classifier_original.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix: \n", metrics.confusion_matrix(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    accuracy.append(metrics.accuracy_score(y_test, y_pred))

    X_train, X_test, y_train, y_test = train_test_split(df2, labels, test_size=0.3)
    rf_classifier_modified = RandomForestClassifier(n_estimators=400)
    rf_classifier_modified.fit(X_train,y_train)
    y_pred = rf_classifier_modified.predict(X_test)

    print('\nModified case\n')

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix: \n", metrics.confusion_matrix(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    recall_modified.append(metrics.recall_score(y_test, y_pred))
    accuracy_modified.append(metrics.accuracy_score(y_test, y_pred))

    filename = '/media/Projects/shared/Classifiers/' + str(IDs[i]) + '_RFclassifier.p'
    pickle.dump(rf_classifier_original, open(filename, 'wb'))

    filename = '/media/Projects/shared/Classifiers/' + str(IDs[i]) + '_RFclassifier_modified.p'
    pickle.dump(rf_classifier_modified, open(filename, 'wb')) 

filename = '/media/Projects/shared/Classifiers/Accuracy.p'
pickle.dump(accuracy, open(filename, 'wb'))

filename = '/media/Projects/shared/Classifiers/Accuracy_modified.p'
pickle.dump(accuracy_modified, open(filename, 'wb'))

filename = '/media/Projects/shared/Classifiers/Recall.p'
pickle.dump(recall, open(filename, 'wb'))

filename = '/media/Projects/shared/Classifiers/Recall_modified.p'
pickle.dump(recall_modified, open(filename, 'wb'))