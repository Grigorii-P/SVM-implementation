import numpy as np
import hand_made_svm as svm
# from hand_made_svm import linear_kernel, polynomial_kernel, gaussian_kernel
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
# import svm_class as svm
from kernels import linear#, polynomial, gaussian

iris = datasets.load_iris()
y = iris.target
sel = [y != 2]
y = y[sel]
X = iris.data[:,:2]  # we only take the first two features.
X = X[sel]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = svm.SVM(kernel=linear)
model.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


xxyy = np.stack((xx.ravel(), yy.ravel()), axis=-1)
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])   ####
Z = model.predict(xxyy)
Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# # Plot also the training points
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.show()