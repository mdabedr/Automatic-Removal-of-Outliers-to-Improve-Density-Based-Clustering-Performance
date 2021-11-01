"""Author Md Abed Rahman: mdabed@cs.ualberta.ca"""
"""This method combines LOF with vanilla HDBSCAN in order to improve HDBSCAN’s performance"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
import timeit
import gc
from hdbscan import HDBSCAN
# import DBCV as DBCV
import seaborn as sns
from scipy.stats import norm
from scipy.stats import zscore

gc.collect()
print(__doc__)

start = timeit.default_timer()

np.random.seed(42)
#Load Data
path = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 697- Into to KDD and DM\\Project\\New datasets'

np.random.seed(42)
filepath = path + '\\DS6\\Tagged.txt'
inputFile = pd.read_csv(filepath, sep='\t')

df = pd.DataFrame([inputFile['X1'], inputFile['X2']])
X = df.transpose()
df = X
X = np.asarray(X, dtype=np.float32)
labels_true = inputFile['Cluster Number']
print(type(labels_true))
outliers_pred=[]
true_pred=[]
extra_X=[]
# fit the model for outlier detection :LOF
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
# Get the true LOF scores
LOF_score = -X_scores

#Plotting code
# gs = gridspec.GridSpec(2, 2)
#
# plt.figure()
#
# ax = plt.subplot(gs[0, 0])  # row 0, col 0
#
# plt.title("Local Outlier Factor (LOF)")
#
# plt.xlim((df['X1'].min() - 100, df['X1'].max() + 100))
# plt.ylim((df['X2'].min() - 100, df['X2'].max() + 100))
#
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# # plot circles with radius proportional to the outlier scores
# radius = (X_scores.min() - X_scores) / (X_scores.min() - X_scores.max())
# plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
#             facecolors='none', label='Outlier scores')
# plt.axis('tight')
#
# legend = plt.legend(loc='upper left')
# legend.legendHandles[0]._sizes = [10]
# legend.legendHandles[1]._sizes = [20]
#
old_max = X_scores.max()
old_min = X_scores.min()

# Find out the index of the points that have an LOF score greater than 2
length = LOF_score.size
flag = list()
for i in range(0, length):
    if LOF_score[i] > 2:
        # print i
        flag.append(i)

#Find percentage of outliers
outlier_percentage=float(len(flag))/float(len(df))*100
print outlier_percentage

count=0
for i in flag:
    if(labels_true[i]!=-1):
        count=count+1

print('no of inliers pruned is '+str(count))

#Prune outliers
df = df.drop(df.index[flag])

LOF_score = np.delete(LOF_score, flag, 0)
X_scores = np.delete(X_scores, flag, 0)
extra_X.extend([X[i] for i in flag])
X = np.delete(X, flag, 0)
print(type(labels_true))
outliers_pred.extend([-1]*len(flag))
true_pred.extend([labels_true[i] for i in flag])
labels_true = np.delete(labels_true.values, flag, 0)

#Plot after outlier pruning
# ax = plt.subplot(gs[0, 1])  # row 0, col 1
# plt.title("Local Outlier Factor (LOF)")
# plt.xlim((df['X1'].min() - 100, df['X1'].max() + 100))
# plt.ylim((df['X2'].min() - 100, df['X2'].max() + 100))
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
#
#
# radius = (old_min - X_scores) / (old_min - old_max)
# plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
#             facecolors='none', label='Outlier scores')
#
# plt.axis('tight')
#
# legend = plt.legend(loc='upper left')
# legend.legendHandles[0]._sizes = [10]
# legend.legendHandles[1]._sizes = [20]

# HDBSCAN Code
X = StandardScaler().fit_transform(X)
hdb = HDBSCAN(min_cluster_size=10,min_samples=5).fit(X)
hdb_labels = hdb.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

print('\n\n++ HDBSCAN Results')
print('Estimated number of clusters: %d' % n_clusters_hdb_)

print('Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, hdb_labels))

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# ax = plt.subplot(gs[1, :])
# Black removed and is used for noise instead.
hdb_unique_labels = set(hdb_labels)

hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(
    hdb_unique_labels)))  # db_colors = plt.cm.Spectral(np.linspace(0, 1, len(db_unique_labels)))# fig = plt.figure(figsize=plt.figaspect(0.5))# hdb_axis = fig.add_subplot('121')# db_axis = fig.add_subplot('122')
for k, col in zip(hdb_unique_labels, hdb_colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    plt.plot(X[hdb_labels == k, 0], X[hdb_labels == k, 1], 'o', markerfacecolor=col, markeredgecolor='k',
             markersize=6)  #

plt.title('LOF-HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)

plt.show()
#True ARI
labels_true=labels_true.tolist()
labels_true.extend(true_pred)
# print(len(labels_true),len()
hdb_labels=hdb_labels.tolist()
hdb_labels.extend(outliers_pred)
X=X.tolist()
X.extend(extra_X)
print('Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, hdb_labels))
print('Mislabeled points: %0.3f'% (np.sum([np.where(np.asarray(labels_true)[i]!=-1,1,0) for i in np.where(np.asarray(hdb_labels)==-1)])+np.sum([np.where(np.asarray(hdb_labels)[i]!=-1,1,0) for i in np.where(np.asarray(labels_true)==-1)])))

filepath=filepath.replace('Tagged.txt','LOF-HDBSCANPoints.txt')

with open(filepath, 'w') as file:
    file.writelines('\t'.join(str(j) for j in i) + '\n' for i in X)

filepath=filepath.replace('Points.txt','cluster.txt')
hdb_labels = [x+1 for x in hdb_labels]
with open(filepath, 'w') as f:
    for _list in hdb_labels:
            #f.seek(0)
        f.write(str(_list)+'\n')


