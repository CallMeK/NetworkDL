import numpy as np

X8_test = np.load('X8_test.npy')
X8_train = np.load('X8_train.npy')
Y_test =np.load('Y_test.npy')
Y_train = np.load('Y_train.npy')
print len(Y_test)

from metric_learn import LMNN


# nca_model = NCA(max_iter=100)
nca_model = LMNN(k=10,min_iter=500,max_iter=1000,learn_rate=1e-4)
nca_model.fit(X8_train[:200],Y_train[:200])
A = nca_model.transformer()

np.fill_diagonal(A,0)

import matplotlib.pyplot as plt

plt.imshow(A)
plt.show()

# import numpy as np
# from metric_learn import NCA
# from sklearn.datasets import load_iris

# iris_data = load_iris()
# X = iris_data['data']
# Y = iris_data['target']
# print Y

# nca = NCA(max_iter=10, learning_rate=0.01)
# nca.fit(X, Y)
# print nca.transformer()
from sklearn.decomposition import PCA
pca2 = PCA(n_components=10)
pc = pca2.fit_transform(A)

ncapc = np.dot(X8_train,pc)

X8_trainS = ncapc
X8_testS = np.dot(X8_test,pc)

print ncapc.shape

color_list = ['ro','bo','ko','y<','g^']
sortedlist = sorted(np.unique(Y_test))
for i in range(X8_test.shape[0]):
    plt.plot(ncapc[i,0],ncapc[i,1],color_list[sortedlist.index(Y_test[i])])
plt.show()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

forest_model = RandomForestClassifier()
forest_model.fit(X8_trainS,Y_train)
print forest_model.score(X8_testS,Y_test)




