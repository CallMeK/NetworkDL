#This file provide the sklearn-like API to SdA.py
import numpy as np
import logging


class DLwrapper:
    def __init__(self,layers=[100],pretraining_epochs=10,training_epochs=200,
                 corruption_value=0.0,batch_size=100,pretrain_lr=0.1,finetune_lr=0.1):
        default_logging_name = "-".join(layers) + str(pretraining_epochs)+"-" + str(training_epochs) + \
                               str(corruption_value) + "-" + str(finetune_lr)
        logging.basicConfig(format='%(asctime)s %(message)s',filename=default_logging_name + '.log', filemode='w', level=logging.DEBUG)
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
        from SdA import train
        self.weights = SdA.train(X,Y,)

    def predict(self,X):
        assert X.shape[0] == Y.shape[0]
        logging.info("Test Start")
        from SdA import predict
        Y_pred = SdA.predict(X)

    def compute_score(self,Y_predict,Y_true):
        return np.sum(Y_predict == Y_true)

    def score(self,X,Y):
        assert X.shape[0] == Y.shape[0]
        logging.info("Score")
        Y_pred = self.predict(X)
