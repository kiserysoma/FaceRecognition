from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import os
import pickle

# Set used parameters
FEATURE_DIFFERENCE = 0.001
TREE_DEPTH = 10
TREE_NUM = 10
SHOW_DECISION_TREE = False
EXTENDED_LOG = False
MODEL_TYPE = 'DT'  # [RF, DT]
DATASET = 'celebA'  # [face, celebA]
if DATASET == 'celebA':
    DATA_PATH = 'celebA_df.pickle'
    PRED_TYPE = 'sex'
elif DATASET == 'face':
    PRED_TYPE = 'ID'  # [ID, Race]
    DATA_PATH = 'face_dataset_embeddings_df_50.p'
    IMG_PATH = 'img_names_50.p'


def evaluate_tree(tree_, print_tree, ext_log):
    log = []  # variable for saving the results

    text_representation = tree.export_text(tree_, decimals=3)
    rows = text_representation.split('\n')

    if print_tree:
        print(text_representation)
        log.append(text_representation + '\n')

    for i in range(len(rows)):

        # Define the current feature name
        if rows[i].count('feature') != 0:
            f_name = str(rows[i].split('re')[1])
            relation = f_name.split(' ')[1]
            if len(f_name.split(' ')[2]) == 0:
                f_value = float(f_name.split(' ')[3])
            else:
                f_value = float(f_name.split(' ')[2])

            f_name = str(f_name.split(' ')[0][1:])
            f_name = str('feature_' + f_name)

            # Find the same feature with same conditions
            for j in range(len(rows)):
                if (i != j) and (rows[j].count(f_name) != 0):
                    if rows[j].split(f_name)[1][0] == ' ':
                        new_row = str(rows[j].split('re')[1])
                        new_relation = str(new_row.split(' ')[1])

                        if len(new_row.split(' ')[2]) == 0:
                            new_value = float(new_row.split(' ')[3])
                        else:
                            new_value = float(new_row.split(' ')[2])

                        # Feature has been found
                        if (abs(new_value - f_value) <= FEATURE_DIFFERENCE) and (new_relation[0] == relation[0]):
                            log.append('\n')
                            log.append('\n')
                            print('\n')
                            print('\n')
                            log.append('--------------------------------------------------------------' + '\n')
                            print('--------------------------------------------------------------')
                            print(f_name)
                            log.append('current feature: ' + f_name + '\n')
                            if ext_log:
                                print('row of the feature:', i)
                                print('row of the finding', j)
                                log.append('row of the feature: ' + str(i) + '\n')
                                log.append('row of the finding: ' + str(j) + '\n')

                            i_depth = rows[i].count('|')
                            j_depth = rows[j].count('|')
                            if ext_log:
                                print("feature depth: ", i_depth)
                                print("finding depth: ", j_depth)
                                log.append("feature depth: " + str(i_depth) + '\n')
                                log.append("finding depth: " + str(j_depth) + '\n')
                                print("depth difference between two features:", abs(i_depth - j_depth))
                                log.append("depth difference between two features: " + str(abs(i_depth - j_depth)) + '\n')
                            try:
                                # Find the prediction leaf node
                                for k in range(i, len(rows)):
                                    if rows[k].count('class') != 0:
                                        to_class_i = k - i
                                        i_depth_class = rows[(i + to_class_i) - 1].count('|') + 1
                                        break

                                # Find previous conditions
                                curr_depth = i_depth
                                cond_i = ''
                                for k in range(i, -1, -1):
                                    if rows[k].count('|') == (curr_depth - 1) and rows[k].count('feature') != 0:
                                        f_ = str(rows[k].split('re')[1])
                                        prev_relation_i = f_.split(' ')[1]
                                        if len(f_.split(' ')[2]) == 0:
                                            f_val = float(f_.split(' ')[3])
                                        else:
                                            f_val = float(f_.split(' ')[2])

                                        f_ = str(f_.split(' ')[0][1:])
                                        f_ = str('feature_' + f_)
                                        curr_depth = rows[k].count('|')
                                        cond_i = str(str(f_) + ' ' + str(prev_relation_i) + ' ' + str(f_val) + ' && ' + cond_i)

                                # Print previous conditions
                                cond_i = cond_i[0:len(cond_i) - 3]
                                print(cond_i)
                                log.append(cond_i + '\n')

                                # Print from the feature to the prediction leaf node
                                for k in range(i, i + to_class_i + 1):
                                    print(rows[k])
                                    log.append(rows[k] + '\n')

                                print('\n')
                                log.append('\n')

                                # Find the prediction leaf node
                                for k in range(j, len(rows)):
                                    if rows[k].count('class') != 0:
                                        to_class_j = k - j
                                        j_depth_class = rows[(j + to_class_j) - 1].count('|') + 1
                                        break

                                # Find previous conditions
                                curr_depth = j_depth
                                cond_j = ''
                                for k in range(j, -1, -1):
                                    if rows[k].count('|') == (curr_depth - 1) and rows[k].count('feature') != 0:
                                        f_ = str(rows[k].split('re')[1])
                                        prev_relation_j = f_.split(' ')[1]
                                        if len(f_.split(' ')[2]) == 0:
                                            f_val = float(f_.split(' ')[3])
                                        else:
                                            f_val = float(f_.split(' ')[2])

                                        f_ = str(f_.split(' ')[0][1:])
                                        f_ = str('feature_' + f_)
                                        curr_depth = rows[k].count('|')
                                        cond_j = str(str(f_) + ' ' + str(prev_relation_j) + ' ' + str(f_val) + ' && ' + cond_j)

                                # Print previous conditions
                                cond_j = cond_j[0:len(cond_j) - 3]
                                print(cond_j)
                                log.append(cond_j + '\n')

                                # Print from the feature to the prediction leaf node
                                for k in range(j, j + to_class_j + 1):
                                    print(rows[k])
                                    log.append(rows[k] + '\n')
                                if ext_log:
                                    diff_class_i = abs(i_depth_class - i_depth)
                                    diff_class_j = abs(j_depth_class - j_depth)

                                    print('depth to prediction leaf node from ' + str(f_name) + ':', diff_class_i)
                                    print('depth to prediction leaf node from ' + str(f_name) + ' found: ', diff_class_j)
                                    log.append('depth to prediction leaf node from ' + str(f_name) + ': ' + str(diff_class_i) + '\n')
                                    log.append('depth to prediction leaf node from ' + str(f_name) + ': ' + str(diff_class_j) + '\n')

                                print('\n')
                                print('\n')
                                print('--------------------------------------------------------------')
                                log.append('\n')
                                log.append('\n')
                                log.append('--------------------------------------------------------------')

                            except:
                                print('error')
    return log


def create_labels(path, pred_type):
    labels = []
    img_names = pickle.load(open(path, 'rb'))
    for i in img_names:
        if PRED_TYPE == 'Race':
            val = i.split("_")[3]
            if val == 'white':
                labels.append(0)
            elif val == 'black':
                labels.append(1)
            elif val == 'indian':
                labels.append(2)
            else:
                labels.append(3)
        else:
            labels.append(i.split('_')[0])
    return labels


def main():

    if DATASET == 'celebA':
        df = pd.read_pickle(DATA_PATH)
        labels = df['target']
        df = df.drop(columns='target')
    elif DATASET == 'face':
        labels = create_labels(IMG_PATH, PRED_TYPE)
        df = pd.read_pickle(DATA_PATH)
    else:
        # Wrong input, do nothing
        print('wrong input')

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3)
    if 'RF' == MODEL_TYPE:
        model = RandomForestClassifier(max_depth=TREE_DEPTH, n_estimators=TREE_NUM)
    elif 'DT' == MODEL_TYPE:
        model = tree.DecisionTreeClassifier(max_depth=TREE_DEPTH)
    else:
        # Wrong input, do nothing
        print('wrong input')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save results
    if not os.path.exists('./LOG/' + DATASET + '/' + PRED_TYPE):
        os.makedirs('./LOG/' + DATASET + '/' + PRED_TYPE)

    # Check decision tree(s)
    if 'RF' == MODEL_TYPE:
        for i_ in range(len(model.estimators_)):
            dt = model.estimators_[i_]
            message = evaluate_tree(dt, SHOW_DECISION_TREE, EXTENDED_LOG)
            out = open('./LOG/' + DATASET + '/' + PRED_TYPE + '/dt_' + str(i_) + '.txt', "w+")
            out.writelines(message)
    elif 'DT' == MODEL_TYPE:
        message = evaluate_tree(model, SHOW_DECISION_TREE, EXTENDED_LOG)
        out = open('./LOG/' + DATASET + '/' + PRED_TYPE + '/DT.txt', "w+")
        out.writelines(message)
    else:
        # Wrong input, do nothing
        print('wrong input')

    # Print model scores
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
    print("Precision:", metrics.precision_score(y_test, y_pred, average="weighted"))
    print("f1:", metrics.f1_score(y_test, y_pred, average="weighted"))


if __name__ == '__main__':
    main()
