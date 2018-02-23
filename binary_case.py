import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kernels import Kernel
import svm
import time
from sklearn import svm as sksvm

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
train = train.drop(['Dates','Date','Time','Resolution','Descript','Address','DayOfWeek', 'WeekOfYear'], axis=1)

cat_cols = ['PdDistrict', 'Year', 'Month', 'Day', 'Hour']
for cat in cat_cols:
    train = pd.get_dummies(train, columns=[cat])

# reduce amount of data for training
train_proportion = 0.02
train_len = len(train)
train = train.sample(frac=1)
train = train.iloc[:int(len(train)*train_proportion)]

# take two the most frequent categories
cats = ['LARCENY/THEFT', 'OTHER OFFENSES']
train_0 = train.loc[train['Category'] == cats[0]]
train_1 = train.loc[train['Category'] == cats[1]]
frames = [train_0, train_1]
data = pd.concat(frames)

# split data for test and training
data = data.as_matrix()
data[data == cats[0]] = 1
data[data == cats[1]] = -1
data = np.array(data, dtype='float')
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0])

print(X_train.shape)
print(X_test.shape)

# training process
start = time.time()
C = 0.1
trainer = svm.SVMTrainer(kernel=Kernel.linear(), c=C)
model = trainer.train(X_train, y_train)
end = time.time()
print(int((end - start) / 60), ' minutes for training')

# predictions
start = time.time()
predictions = []
for x in X_test:
    predictions.append(model.predict(x))
end = time.time()
print(int((end - start) / 60), ' minutes for predicting')

# scikit SVM training
clf = sksvm.SVC()
clf.fit(X_train, y_train)
pred_real_svm = clf.predict(X_test)

# compare results
print('my accuracy is ', accuracy(y_test, predictions))
print('scikit accuracy is ', accuracy(y_test, pred_real_svm))