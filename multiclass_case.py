import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kernels import Kernel
import svm
import time
import copy

def accuracy(x, y):
    count = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            count += 1
    return count/len(x)


# break down the 'Date' field and make one-hot vectors out of categorical fields
train = pd.read_csv('/Users/grigoriipogorelov/Desktop/train.csv')
train['Date'] = pd.to_datetime(train['Dates'], errors='coerce')
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['WeekOfYear'] = train['Date'].dt.weekofyear
train['Day'] = train['Date'].dt.day
train['Time'] = train['Date'].dt.time
train['Hour'] = train['Date'].dt.hour
train = train.drop(['Dates','Date','Time','Resolution','Descript','Address','WeekOfYear', 'DayOfWeek'], axis=1)

cat_cols = ['PdDistrict', 'Year', 'Month', 'Day', 'Hour']
for cat in cat_cols:
    train = pd.get_dummies(train, columns=[cat])

# reduce amount of data for training
train_len = len(train)
data_portion = 0.008
# shuffle the data
train = train.sample(frac=1)
train = train.iloc[:int(len(train)*data_portion)]
copy_ = copy.copy(train)
test = copy_.iloc[int(len(train)*0.8):]
train = train.iloc[:int(len(train)*0.8)]

# samples - paired categories
# For example, if there are three categories - THEFT, ASSAULT, BURGLARY,
# then samples would be: [THEFT+ASSAULT, THEFT+BURGLARY, ASSAULT+BURGLARY]
# This is for training (n(n+1)/2) classifiers
categories = train.Category.unique()
category_len = len(train['Category'].unique())
samples = []
for i in range(category_len - 1):
    for j in range(i + 1, category_len):
        samples.append(categories[i] + '+' + categories[j])

# breaking down the dataset for getting (n(n+1)/2) pieces according to the number of classifiers
train_sets = []
for sample in samples:
    cats = sample.split('+')
    train_0 = train.loc[train['Category'] == cats[0]]
    train_1 = train.loc[train['Category'] == cats[1]]
    frames = [train_0, train_1]
    train_sets.append(pd.concat(frames))
print(train.shape)

# training process
start = time.time()
models = []
cat_to_number_each_model = []
C = 0.1
for i, data in enumerate(train_sets):
    _d = {}
    options = data['Category'].unique()
    data = data.as_matrix()
    data[data == options[0]] = 1
    _d['1'] = options[0]
    data[data == options[1]] = -1
    _d['-1'] = options[1]
    data = np.array(data, dtype='float')
    trainer = svm.SVMTrainer(kernel=Kernel.linear(), c=C)
    model = trainer.train(data[:,1:], data[:,0])
    models.append(model)
    cat_to_number_each_model.append(_d)
    print()
    print('MODEL '+str(i)+' finished')
    print()
end = time.time()
print(int((end - start)/60), ' minutes for training')

labels = test['Category'].as_matrix()
test = test.as_matrix()
test = np.array(test[:, 1:], dtype='float')

# predictions for the whole dataset for each of the models
start = time.time()
predictions_for_each_model = []
for model in models:
    pred = []
    for t in test:
        pred.append(model.predict(t))
    predictions_for_each_model.append(pred)

end = time.time()
print(int((end - start) / 60), ' minutes for predicting')

count_for_categories = {}
for cat in categories:
    count_for_categories[cat] = 0

# here we have a major voting, predictions from all the models for a sample item are combined,
# the most frequent category wins
combo_predictions = []
pred_len = len(predictions_for_each_model[0])
# for every prediction of a model
for i in range(pred_len):
    count_cat = copy.deepcopy(count_for_categories)
    # take a particular model
    for j, preds in enumerate(predictions_for_each_model):
        count_cat[cat_to_number_each_model[j][str(int(preds[i]))]] += 1
    winner = max(count_cat, key=count_cat.get)
    combo_predictions.append(winner)

accuracy = accuracy(labels, combo_predictions)
print('accuracy is ', accuracy)

# one of the major problems lies in 'cvxopt', cause it's calculations are too time-consuming
