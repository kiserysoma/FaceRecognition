from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
import random
import matplotlib.pyplot as plt
from evaluateDTs import create_labels
import os

CLF = 'RF'  # [RF, DT]
DATASET = 'face'  # [face, celebA]
if DATASET == 'celebA':
    DATA_PATH = 'celebA_df.pickle'
    PRED_TYPE = 'sex'
elif DATASET == 'face':
    PRED_TYPE = 'ID'  # [ID, race]
    DATA_PATH = 'face_dataset_embeddings_df_50.p'
    IMG_PATH = 'img_names_50.p'

accuracies = []
recalls = []
precisions = []
f1s = []
features = []
labels = []

if DATASET == 'face':
    labels = create_labels(IMG_PATH, PRED_TYPE)
    df = pd.read_pickle(DATA_PATH)
elif DATASET == 'celebA':
    df = pd.read_pickle(DATA_PATH)
    labels = df['target']
    df = df.drop(columns='target')
else:
    # Wrong input, do nothing
    print('wrong input')

for i in tqdm(range(0, 128, 10)):
    if i != 0:
        col_to_drop = list(random.sample(list(df.columns), i))
        #print('removed features:', col_to_drop)
        df2 = df.drop(columns=col_to_drop)
    else:
        df2 = df
    features.append(i + 1)

    # Train selected model
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(df2, labels, test_size=0.3)
    if CLF == 'RF':
        model = RandomForestClassifier(n_estimators=100, max_depth=15)
    elif CLF == "DT":
        model = tree.DecisionTreeClassifier(max_depth=10)
    else:
        print('wrong model type')
    model.fit(X_train_mod, y_train_mod)
    y_pred_mod = model.predict(X_test_mod)

    accuracies.append(metrics.accuracy_score(y_test_mod, y_pred_mod))
    recalls.append(metrics.recall_score(y_test_mod, y_pred_mod, average="weighted"))
    f1s.append(metrics.f1_score(y_test_mod, y_pred_mod, average="weighted"))

plt.plot(features, accuracies, color="red", label="accuracy")
plt.plot(features, recalls, color="blue", label="recall")
plt.plot(features, f1s, color="black", label="f1")

if CLF == 'RF':
    m_type = 'Random Forest'
elif CLF == "DT":
    m_type = 'Decision Tree'
else:
    print('wrong model type')

plt.title("Used classifier: " + m_type + ", Prediction type: " + PRED_TYPE + ', Dataset: ' + DATASET)
plt.ylabel('Accuracy, precision, recall, f1', fontsize=16)
plt.xlabel('Number of features removed', fontsize=16)
plt.legend(loc="best")

if not os.path.exists('./Plots'):
    os.makedirs('./Plots')
plt.savefig('./Plots/metrics_random_' + CLF + '_' + PRED_TYPE + '_' + DATASET + '.png')
plt.show()
