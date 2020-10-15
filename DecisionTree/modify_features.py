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
NOISE = 'random'  # ['0', 'random', 'multiplicative', 'additive']
MOD = 'ImportantFeatures'
important_features_sex = ['f83', 'f41', 'f10', 'f84', 'f14', 'f53', 'f43', 'f42', 'f76', 'f102']
#important_features_race = ['f83', 'f58', 'f81', 'f48', 'f55', 'f37', 'f89', 'f86', 'f49', 'f50']
important_features_race = [82, 57, 80, 47, 54, 36, 88, 85, 48, 49, 20, 35, 34, 10]


def average(lst):
    return sum(lst) / len(lst)


def saturate_df_values(df):
    ret_df = df
    ret_df = ret_df.clip(-1, 1)
    return ret_df


def split_face_df(df, labels):
    train_labels = []
    test_labels = []
    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)
    train_rows = 0
    test_rows = 0

    for i in range(len(labels)):
        if (i % 3) == 0:
            test_labels.append(labels[i])
        else:
            train_labels.append(labels[i])

    for i in range(len(df.index)):
        if (i % 3) == 0:
            test_data.loc[test_rows] = df.iloc[i]
            test_rows += 1
        else:
            train_data.loc[train_rows] = df.iloc[i]
            train_rows += 1

    return train_data, train_labels, test_data, test_labels


def train(model_type, train_data, train_labels, test_data, test_labels):
    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=50, max_depth=10)
    elif model_type == "DT":
        model = tree.DecisionTreeClassifier(max_depth=10)
    else:
        print('wrong model type')
    model.fit(train_data, train_labels)
    y_pred = model.predict(test_data)
    print(metrics.accuracy_score(test_labels, y_pred))
    print(metrics.recall_score(test_labels, y_pred, average="weighted"))
    print(metrics.f1_score(test_labels, y_pred, average="weighted"))
    return model


def replace_0s(df):
    ret_df = df
    for f in important_features_race:
        ret_df[f] = 0
    return ret_df


def replace_random(df):
    ret_df = df
    rnd = []
    for f in important_features_race:
        num = random.uniform(-1, 1)
        ret_df[f] = num
        rnd.append(num)
    return ret_df, rnd


def add_m_noise(df):
    ret_df = df
    for f in important_features_race:
        r = random.randint(-10, 10)
        ret_df[f] = r*ret_df[f]
    ret_df = saturate_df_values(ret_df)
    return ret_df


def add_a_noise(df, factor):
    ret_df = df
    for f in important_features_race:
        random_value = random.uniform(-1, 1)
        ret_df[f] = (random_value * factor) + ret_df[f]
    ret_df = saturate_df_values(ret_df)
    return ret_df


if DATASET == 'celebA':
    df = pd.read_pickle(DATA_PATH)
    labels = df['target']
    df = df.drop(columns='target')
elif DATASET == 'face':
    labels = create_labels(IMG_PATH, PRED_TYPE)
    df = pd.read_pickle(DATA_PATH)
    train_df, train_labels, test_df, test_labels = split_face_df(df, labels)
else:
    print('wrong input')

#train face recognition model
print('face recognition')
face_model = train(CLF, train_df, train_labels, test_df, test_labels)

#train race predictor
print('race prediction')
r_labels = create_labels(IMG_PATH, 'race')
_, train_r_labels, __, test_r_labels = train_df, train_labels, test_df, test_labels = split_face_df(df, r_labels)
race_model = train(CLF, train_df, train_r_labels, test_df, test_r_labels)

if NOISE != 'additive':
    max_diff = 0
    face = 0
    race = 0
    face_scores = []
    race_scores = []
    rnd_removed_features_face = []  # model average scores after random features removed
    rnd_removed_features_race = []
    modified_features = []

    for i in tqdm(range(1, 128, 10)):
        if MOD == 'Mod_rnd_features':
            important_features_race = list(random.sample(list(train_df.columns), i))
            modified_features.append(i)
        for _ in range(200):
            if NOISE == 'random':
                new_df, _ = replace_random(test_df)
            elif NOISE == '0':
                new_df = replace_0s(test_df)
            else:
                new_df = add_m_noise(test_df)
            try:
                y_pred = face_model.predict(new_df)
                acc_face = metrics.accuracy_score(test_labels, y_pred)
                face_scores.append(acc_face)
                y_pred = race_model.predict(new_df)
                acc_race = metrics.accuracy_score(test_r_labels, y_pred)
                race_scores.append(acc_race)
                diff = acc_face - acc_race
                if max_diff < diff:
                    face = acc_face
                    race = acc_race
            except:
                continue
        rnd_removed_features_face.append(average(face_scores))
        rnd_removed_features_race.append(average(race_scores))

    if MOD == 'Mod_rnd_features':
        plt.plot(modified_features, rnd_removed_features_face, color="red", label="avg. face rec.")
        plt.plot(modified_features, rnd_removed_features_race, color="blue", label="avg. race pred.")
        plt.title("Average scores")
        plt.ylabel('Accuracy',  fontsize=16)
        plt.xlabel('Number of modified features', fontsize=16)
        plt.legend(loc="best")

        if not os.path.exists('./Plots/Modified'):
            os.makedirs('./Plots/Modified')
        plt.savefig('./Plots/Modified/''avg_acc_method:' + NOISE + '_' + DATASET + '_' + MOD + '.png')
        plt.show()

    else:
        print(average(rnd_removed_features_face))
        print(average(rnd_removed_features_race))

else:
    factors = [0.001, 0.01, 0.1, 0.2, 0.3, 0.35]
    faces_max = []
    races_max = []
    faces = []
    races = []
    avg_face = [0.90]
    avg_race = [0.90]
    iterations = [0]
    for i in tqdm(range(len(factors))):
        max_diff = 0
        iterations.append(i+1)
        for _ in range(1000):
            new_df = add_a_noise(test_df, factors[i])
            try:
                y_pred = face_model.predict(new_df)
                acc_face = metrics.accuracy_score(test_labels, y_pred)
                faces.append(acc_face)
                y_pred = race_model.predict(new_df)
                acc_race = metrics.accuracy_score(test_r_labels, y_pred)
                races.append(acc_race)

                #Calculate max, average
                diff = acc_face - acc_race
                if max_diff < diff:
                    face_m = acc_face
                    race_m = acc_race
            except:
                continue

        avg_face.append(average(faces))
        avg_race.append(average(races))
        faces_max.append(face_m)
        races_max.append(race_m)

    plt.plot(iterations, avg_face, color="red", label="avg. face rec.")
    plt.plot(iterations, avg_race, color="blue", label="avg. race pred.")
    #plt.xticks(factors)
    plt.title("Average scores")
    plt.ylabel('Accuracy',  fontsize=16)
    plt.xlabel('Factors', fontsize=16)
    plt.legend(loc="best")

    if not os.path.exists('./Plots/Modified'):
        os.makedirs('./Plots/Modified')
    plt.savefig('./Plots/Modified/''avg_acc_method:' + NOISE + '_' + DATASET + '_' + MOD + '.png')
    plt.show()
