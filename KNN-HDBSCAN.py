"""Author Md Abed Rahman: mdabed@cs.ualberta.ca"""
"""This method combines KNN with vanilla HDBSCAN in order to improve HDBSCAN’s performance"""
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
from pyod.models.knn import KNN
gc.collect()
print(__doc__)

start = timeit.default_timer()

np.random.seed(42)
#Load Data
path='G:\\Poralekha\\UofA\Fall 2019\\CMPUT 697- Into to KDD and DM\\Project\\New datasets'
np.random.seed(42)
filepath=path+'\\DS6\\Tagged.txt'
inputFile =  pd.read_csv(filepath, sep='\t')
df = pd.DataFrame([inputFile['X1'], inputFile['X2']])
X = df.transpose()
df = X
X = np.asarray(X, dtype=np.float32)

labels_true=inputFile['Cluster Number']
outliers_pred=[]
true_pred=[]
extra_X=[]

#KNN based outlier detection

knn = KNN(n_neighbors=20, contamination=0.1, method='largest')
knn.fit(X)
# get the prediction labels and outlier scores of the training data
y_pred = knn.labels_
X_scores = knn.decision_scores_
LOF_score = X_scores
#Code for plotting

# gs = gridspec.GridSpec(2, 2)
# plt.figure()
# ax = plt.subplot(gs[0, 0])  # row 0, col 0
# plt.title("KNN distance based outlier")
# plt.xlim((df['X1'].min()-100, df['X1'].max()+100))
# plt.ylim((df['X2'].min()-100, df['X2'].max()+100))
#
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
#
# # plot circles with radius proportional to the outlier scores
# radius = (X_scores.min() - X_scores) / (X_scores.min() - X_scores.max())
# plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
#             facecolors='none', label='Outlier scores')
# plt.axis('tight')
#
# legend = plt.legend(loc='upper left')
# legend.legendHandles[0]._sizes = [10]
# legend.legendHandles[1]._sizes = [20]

old_max = X_scores.max()
old_min = X_scores.min()

#Find out the index of the points that are deemed as outliers by KNN
length = y_pred.size
flag = list()
for i in range(0, length):
    if y_pred[i] == 1:
        flag.append(i)

count=0
for i in flag:
    if(labels_true[i]!=-1):
        count=count+1

print('no of inliers pruned is '+str(count))
#Prune inliers
df = df.drop(df.index[flag])

LOF_score = np.delete(LOF_score, flag, 0)
X_scores = np.delete(X_scores, flag, 0)
extra_X.extend([X[i] for i in flag])
X = np.delete(X, flag, 0)
outliers_pred.extend([-1]*len(flag))
true_pred.extend([labels_true[i] for i in flag])
labels_true= np.delete(labels_true.values,flag,0)

#Plot results after pruning
# ax = plt.subplot(gs[0, 1])  # row 0, col 1
# plt.title("KNN distance based outlier")
# plt.xlim((df['X1'].min()-100, df['X1'].max()+100))
# plt.ylim((df['X2'].min()-100, df['X2'].max()+100))
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# # plot circles with radius proportional to the outlier scores
# radius = (old_min- X_scores) / (old_min - old_max)
# #radius = (old_max - X_scores) / (old_max - old_min)
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
print(hdb_labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)


#Printing number of clusters, ARI and SWC
print('\n\n++KNN-HDBSCAN Results')
print('Estimated number of clusters: %d' % n_clusters_hdb_)
print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, hdb_labels))



##############################################################################


# Plot result
import matplotlib.pyplot as plt
# ax = plt.subplot(gs[1, :])
# Black removed and is used for noise instead.
hdb_unique_labels = set(hdb_labels)

hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))

for k, col in zip(hdb_unique_labels, hdb_colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    plt.plot(X[hdb_labels == k, 0], X[hdb_labels == k, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)


plt.title('KNN-HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)

plt.show()
#True ARI
labels_true=labels_true.tolist()
labels_true.extend(true_pred)
hdb_labels=hdb_labels.tolist()
hdb_labels.extend(outliers_pred)
X=X.tolist()
X.extend(extra_X)

print('Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, hdb_labels))
print('Mislabeled points: %0.3f'% (np.sum([np.where(np.asarray(labels_true)[i]!=-1,1,0) for i in np.where(np.asarray(hdb_labels)==-1)])+np.sum([np.where(np.asarray(hdb_labels)[i]!=-1,1,0) for i in np.where(np.asarray(labels_true)==-1)])))


# with open('output.txt', 'w') as f:
#     for _list in X:
#         for _string1,_string2 in _list:
#             #f.seek(0)
#             f.write(str(_string1)+ str(_string2)+ '\n')

filepath=filepath.replace('Tagged.txt','KNN-HDBSCANPoints.txt')

with open(filepath, 'w') as file:
    file.writelines('\t'.join(str(j) for j in i) + '\n' for i in X)

filepath=filepath.replace('Points.txt','cluster.txt')
hdb_labels = [x+1 for x in hdb_labels]
print(len(hdb_labels))
print(hdb_labels)
with open(filepath, 'w') as f:
    for _list in hdb_labels:
            #f.seek(0)
        f.write(str(_list)+'\n')