# This is a experiment script for computing the scores for one of the three CV result


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd

class Experiment(object):
    def __init__(self,modelobj,X,Y):
        self.modelobj = modelobj
        self.X = X
        self.Y = Y

    def CV_scores(self,cv=5):
        return cross_val_score(self.modelobj, self.X, self.Y, cv=cv)

data_folder_dir = "../Extension/AdjmatData/"
report = pd.DataFrame(columns = ['Nodes','FeatureType','Method','Score','Error'])

print "Logistic Regression, image"

row_id = 0
lr_model = LR(C=1.0)
rf_model = RF(500)

methods_lib = {'LR':lr_model,'RF':rf_model}


for method in methods_lib:
    for d in [8,16,32]:
        X = np.load(data_folder_dir + ("/%d/" %d) + 'X_data_image.npy')
        Y = np.load(data_folder_dir + ("/%d/" %d) + 'Y_data.npy')

        modelobj = methods_lib[method]
        experiment = Experiment(modelobj, X, Y)

        scores = experiment.CV_scores()
        score_mean =  np.mean(scores)
        score_std = np.std(scores)
        print score_mean
        print score_std
        report.loc[row_id] = [d, 'image', method, "%.3f" %score_mean, "%.3f" %score_std]
        row_id = row_id + 1

report.to_csv('results/LR_RF.csv')



