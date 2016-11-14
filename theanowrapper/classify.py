import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import sys


# X = np.load("X_stats.npy")
# X = np.load("X_resized_8.npy")
# X = np.load("X_resized_16.npy")
X = np.load("X_16_and_32_resized_to_32.npy")
Y = np.load("Y_data.npy")


from sklearn.cross_validation import train_test_split


for i in xrange(len(Y)):
	Y[i] = Y[i].split('_')[0]


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=12345)




print np.unique(Y)


np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)


sys.exit()
# Note that first 8 columns are classic features, remaining 1024 are "image" features

log_model = LogisticRegression()
forest_model = RandomForestClassifier(n_estimators=500)

# # LR on stats
# log_scores_classic = cross_validation.cross_val_score(log_model, X, Y, cv=10)  # 10 fold cross validation
# print "Logistic Regression on Classic Features:"
# print np.average(log_scores_classic)

# # RF on stats
# average_score = 0
# for i in range(NUM_ITERS):
#   forest_scores_classic = cross_validation.cross_val_score(forest_model, X, Y, cv=10)  # 10 fold cross validation
#   score = np.average(forest_scores_classic)
#   average_score += score / float(NUM_ITERS)
# print "Randomized Forest on Classic Features:"
# print average_score

# LR on image
log_scores_image = cross_validation.cross_val_score(log_model, X, Y, cv=10)  # 10 fold cross validation
print "Logistic Regression on Image Features:"
print np.average(log_scores_image)

# RF on image

forest_scores_image = cross_validation.cross_val_score(forest_model, X, Y, cv=10)  # 10 fold cross validation
score = np.average(forest_scores_image)

print "Randomized Forest on Image Features:"
print score

