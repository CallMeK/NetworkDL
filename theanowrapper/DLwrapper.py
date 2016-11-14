#This file provide the sklearn-like API to SdA.py
import numpy as np
import logging
import NSdA
from sklearn.base import BaseEstimator, ClassifierMixin

class DLwrapper(BaseEstimator, ClassifierMixin):

    VALIDATION_RATIO = 0.1

    def __init__(self,layers=[100],pretraining_epochs=10,training_epochs=200,
                 corruption_value=0.0,batch_size=100,pretrain_lr=0.1,finetune_lr=0.1):
        default_logging_name = "-".join([str(l) for l in layers]) + "-"+ str(pretraining_epochs)+"-" + str(training_epochs) + "-" + \
                               str(corruption_value) + "-" + str(finetune_lr)
        logging.basicConfig(format='%(asctime)s %(message)s',filename="logs/" + default_logging_name + '.log', filemode='w', level=logging.DEBUG)
        self.layers = layers
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.corruption_value = corruption_value
        self.batch_size = batch_size
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr

    def fit(self,X,Y):
        assert X.shape[0] == Y.shape[0]
        self.input_dim = X.shape[1]
        self.training_N = X.shape[0]
        logging.info("Training Start")
        self.sda, self.labels = NSdA.train(X,Y,network=self.layers,finetune_lr=self.finetune_lr,
              pretraining_epochs=self.pretraining_epochs,
              pretrain_lr=self.pretrain_lr,
              training_epochs=self.training_epochs,
              batch_size=self.batch_size,
              corruption_value=self.corruption_value,
              validation_ratio=self.VALIDATION_RATIO)
        self.labels = np.array(self.labels)

    def predict(self,X):
        logging.info("Test Start")
        _Y_pred = NSdA.predict(X,self.sda)
        Y_pred = self.labels[_Y_pred]
        return Y_pred

    def compute_score(self,Y_predict,Y_true):
        return np.sum(Y_predict == Y_true)/float(Y_predict.shape[0])

    def score(self,X,Y):
        assert X.shape[0] == Y.shape[0]
        logging.info("Score")
        Y_pred = self.predict(X)
        return self.compute_score(Y_pred,Y)


if __name__ == "__main__":
    model = DLwrapper()
    X_8 = np.load("../Extension/AdjmatData/8/X_data_image.npy")
    Y_8 = np.load("../Extension/AdjmatData/8/Y_data.npy")
    model.fit(X_8,Y_8)
    Y_pred = model.predict(X_8)
    print model.compute_score(Y_pred,Y_8)



