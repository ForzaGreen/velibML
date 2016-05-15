# coding=utf-8

import numpy as np
import matplotlib.dates as md
from sklearn import linear_model
import matplotlib.pyplot as plt

data = np.loadtxt(fname="data.csv", delimiter=',', converters = {0: md.datestr2num}, skiprows=1)
# I will use 70% of data for training, 30% for test
m = data.shape[0] # total nbr of examples

X = data[0 : int(m*0.7), 0:9]
y = data[0 : int(m*0.7), 11]

X_test = data[int(m*0.7) + 1: m-1 , 0:9]
y_test = data[int(m*0.7) + 1: m-1 , 11]

clf = linear_model.LinearRegression()
clf.fit (X, y)

print "coef: ", clf.coef_
print "intercept: ", clf.intercept_ # theta_zero
#print "params used: ", clf.get_params()
print "score: ", clf.score(X_test, y_test)

###############################################################################
# Compute paths

n_alphas = 200 # Andrew appelle ça lambda
alphas = np.logspace(-2, 5, n_alphas)
alphas = np.insert(alphas, 0, 0) # zero element corresponds to linear regression without regularisation

clf = linear_model.Ridge()

scores = []

for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    scores.append(clf.score(X_test, y_test))
    #print a, clf.score(X_test, y_test)

print "Min score: ", np.min(scores)
print "Max score: ", np.max(scores)
print "\nbest alpha: ", alphas[np.argmax(scores)], " score: ", np.max(scores)

###############################################################################
# Display results

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, scores)
ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim())
plt.xlabel('alpha')
plt.ylabel('score')
plt.title('score as a function of the regularization')
plt.axis('tight')
plt.show()

