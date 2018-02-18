import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import svm
from kernels import Kernel

iris = datasets.load_iris()
y = iris.target
y = np.asarray([-1 if x == 0 else x for x in y])
sel = [y != 2]
y = y[sel]
X = iris.data[:,:2]
X = X[sel]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

C = 0.1
trainer = svm.SVMTrainer(kernel=Kernel.linear(), c=C)
model = trainer.train(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


xxyy = np.stack((xx.ravel(), yy.ravel()), axis=-1)

result = []
for i in range(len(xxyy)):
    result.append(model.predict(xxyy[i]))

Z = np.array(result).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()