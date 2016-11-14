# This is a experiment script for computing the scores for one of the three CV result


import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
from theanowrapper.DLwrapper import DLwrapper

class Experiment(object):
    def __init__(self,modelobj,X,Y):
        self.modelobj = modelobj
        self.X = X
        self.Y = Y

    def CV_scores(self,cv=5):
        return cross_val_score(self.modelobj, self.X, self.Y, cv=cv)

data_folder_dir = "../Extension/AdjmatData/"
report = pd.DataFrame(columns = ['Nodes','FeatureType','Method','Score','Error'])

print "Deep Learning, image"

row_id = 0


for method in ['DL']:
    for d in [8,16,32]:
        X = np.load(data_folder_dir + ("/%d/" %d) + 'X_data_image.npy')
        Y = np.load(data_folder_dir + ("/%d/" %d) + 'Y_data.npy')

        modelobj = DLwrapper(layers=[20,15], pretraining_epochs=10, training_epochs=400,batch_size=100,pretrain_lr=0.001,finetune_lr=0.2)
        experiment = Experiment(modelobj, X, Y)

        scores = experiment.CV_scores()
        score_mean =  np.mean(scores)
        score_std = np.std(scores)
        print score_mean
        print score_std
        report.loc[row_id] = [d, 'image', method, "%.3f" %score_mean, "%.3f" %score_std]
        row_id = row_id + 1

report.to_csv('results/DL_paper.csv')



