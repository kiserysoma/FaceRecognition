import sklearn
import lime.lime_tabular
import pickle
import pandas as pd
import numpy as np

def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

def get_prediction(result_list):
  max_pred = 0
  index = 0
  race = 0
  for i in result_list:
    if i[0][1] > max_pred:
      max_pred = i[0][1]
      race = index
    index += 1
  return race

def get_race(model, instance):
  result_list = model.predict_proba(instance)
  prediction = []
  for i in result_list:
    prediction.append((i[0][1]))
  return prediction

FEATURES_TO_OBSERVE = 10

train_x = pd.read_pickle('Race_x_train.pickle')
train_y = pd.read_pickle('Race_y_train.pickle')
test_x = pd.read_pickle('Race_x_test.pickle')
test_y = pd.read_pickle('Race_y_test.pickle')
filename = 'Race_model.pickle'
rf_model = pickle.load(open(filename, 'rb'))

unique(test_y)

lst = rf_model.predict_proba(test_x.loc[[2957]])
print(get_predicted_race(lst))

predict_fn_rf = lambda x: rf_model.predict_proba(x).astype('float')

#create explainer object
#class names will be changed to races
explainer = lime.lime_tabular.LimeTabularExplainer(train_x.values,feature_names = train_x.columns,class_names=['0','1','2','3'],kernel_width=5)

c = 0
l = []
indexes = test_x.index.values
for i in indexes[0:1000]:
  print(c)
  choosen_instance = test_x.loc[[i]].values[0]
  lst = rf_model.predict_proba(test_x.loc[[i]])

  #predict_fn_rf can't be used
  exp = explainer.explain_instance(choosen_instance, predict_fn_rf ,num_features=FEATURES_TO_OBSERVE)
  a = exp.as_list()
  l.append(a)
  c += 1

pickle.dump( l, open( "predictionRace.p", "wb" ) )
