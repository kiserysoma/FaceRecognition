import sklearn
import lime.lime_tabular
import pickle
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

FEATURES_TO_OBSERVE = 10
filename = 'Sex_model.pickle'
rf_model = pickle.load(open(filename, 'rb'))

train_x = pd.read_pickle('Sex_x_train.pickle')
train_y = pd.read_pickle('Sex_y_train.pickle')
test_x = pd.read_pickle('Sex_x_test.pickle')
test_y = pd.read_pickle('Sex_y_test.pickle')

predict_fn_rf = lambda x: rf_model.predict_proba(x).astype(float)
#create explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(train_x.values,feature_names = train_x.columns,class_names=['Male','Female'],kernel_width=5)

c = 0
male = []
female = []
l = []

indexes = test_x.index.values
for i in indexes[0:1000]:
  print(c)
  choosen_instance = test_x.loc[[i]].values[0]
  exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=FEATURES_TO_OBSERVE)
  a = exp.as_list()
  l.append(a)
  if int(test_y.loc[[i]].values[0]) == 0:
    print("m")
    male.append(a)
  elif int(test_y.loc[[i]].values[0]) == 1:
    print("f")
    female.append(a)

  print(i)
  c += 1
print(len(l))

pickle.dump( l, open( "prediction.p", "wb" ) )
pickle.dump( male, open( "predictionM.p", "wb" ) )
pickle.dump( female, open( "predictionF.p", "wb" ) )

filename = 'prediction.p'
predictions = pickle.load(open(filename, 'rb'))

filename = 'predictionF.p'
predictionsF = pickle.load(open(filename, 'rb'))

filename = 'predictionM.p'
predictionsM = pickle.load(open(filename, 'rb'))

def process_predictions(predictions):
  data_elements = []
  for i in range(len(predictions)):
    for j in range(len(predictions[i])):

      index = predictions[i][j][0].find('f')
      space = predictions[i][j][0][index:index+5].find(' ')
      data_elements.append(predictions[i][j][0][index:(index + space)])

  return data_elements

predF = process_predictions(predictionsF)
predM = process_predictions(predictionsM)
pred  = process_predictions(predictions)


def visualize_results(results):
  x = []
  y = []
  most_common = [item for item in Counter(results).most_common()]
  for i in range(len(most_common)):
    x.append(most_common[i][0])
    y.append(most_common[i][1])

  plt.bar(x, y, color='red')
  plt.xlabel("Features")
  plt.ylabel("Count")
  plt.title("Feature distribution")
  plt.show()
  print('Most common features:', most_common[0][0],most_common[1][0],most_common[2][0])

visualize_results(pred)
visualize_results(predF)
visualize_results(predM)