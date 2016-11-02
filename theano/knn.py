from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np 
import sys

X_train = np.load('training_final.npy')
X_test = np.load('test_final.npy')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')

Y_trainExta = np.load('../statsv2_and_resized_graphs/3216padding/Y_trainT.npy')

print sum(Y_train==Y_trainExta)


sys.exit()
print sorted(np.unique(Y_test))

knn = KNN(n_neighbors=13)
knn.fit(X_train,Y_train)
print knn.score(X_test,Y_test)

from sklearn.decomposition import PCA, KernelPCA
pca = PCA(n_components=17)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)

knn = KNN(n_neighbors=17)
knn.fit(X_train1,Y_train)
print knn.score(X_test1,Y_test)

# kpca = KernelPCA(n_components=100, kernel = 'rbf',eigen_solver='arpack')
# X_train2 = kpca.fit_transform(X_train)
# X_test2 = pca.transform(X_test)

# knn = KNN(n_neighbors=15)
# knn.fit(X_train2,Y_train)
# print knn.score(X_test2,Y_test)

sys.exit()

from metric_learn import LMNN,NCA


# nca_model = NCA(max_iter=100)
nca_model = LMNN(k=13,min_iter=100,max_iter=500,learn_rate=1e-2)
nca_model.fit(X_train[:100],Y_train[:100])
A = nca_model.transformer()

# np.fill_diagonal(A,0)

import matplotlib.pyplot as plt

plt.imshow(A)
plt.show()


from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pc = pca2.fit_transform(A)


X_trainS = np.dot(X_train,pc)
X_testS = np.dot(X_test,pc)

X_trainFull = np.dot(X_train,A)
X_testFull = np.dot(X_test,A)


color_list = ['ro','bo','ko','y<','g^']
sortedlist = sorted(np.unique(Y_test))

print sortedlist.index(Y_test[0])

for i in range(X_testS.shape[0]):
    plt.plot(X_testS[i,0],X_testS[i,1],color_list[sortedlist.index(Y_test[i])])
plt.show()

knn = KNN(n_neighbors=13)
knn.fit(X_trainFull,Y_train)
print knn.score(X_testFull,Y_test)

from sklearn.linear_model import LogisticRegression as LR
lr = LR()
lr.fit(X_trainFull,Y_train)
print lr.score(X_testFull,Y_test)




