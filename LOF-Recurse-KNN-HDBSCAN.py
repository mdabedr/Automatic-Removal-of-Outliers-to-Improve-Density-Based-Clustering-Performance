"""Author Md Abed Rahman: mdabed@cs.ualberta.ca"""
"""This method recurses for LOF until approximated outlier percentage goes under a threshold; at which point it only runs KNN once to get rid
of the global outliers. This method is less prone to removing inliers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
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

path = 'G:\\Poralekha\\UofA\Fall 2019\\CMPUT 697- Into to KDD and DM\\Project\\New datasets'

np.random.seed(42)
filepath = path + '\\DS6\\Tagged.txt'
inputFile = pd.read_csv(filepath, sep='\t')

# Read dataset
df = pd.DataFrame([inputFile['X1'], inputFile['X2']])
X = df.transpose()
df = X
X = np.asarray(X, dtype=np.float32)
labels_true = inputFile['Cluster Number']
outliers_pred=[]
true_pred=[]
extra_X=[]

print(type(labels_true))

# Make plot grid of 2X2
# gs = gridspec.GridSpec(2, 2)
#
# plt.figure()
#
# ax = plt.subplot(gs[0, 0])  # row 0, col 0
#
counter = 0
percent=[]
while (True):

    # fit the model for outlier detection with LOF
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_  # Default model has negative of LOF scores
    # Get the true LOF scores
    LOF_score = -X_scores

    if counter == 0:
        # Plot with LOF
        # plt.title("Local Outlier Factor (LOF)")
        # # plt.subplot(211)
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
        # # plt.xlabel("prediction errors: %d" % (n_errors))
        # legend = plt.legend(loc='upper left')
        # legend.legendHandles[0]._sizes = [10]
        # legend.legendHandles[1]._sizes = [20]

        old_max = X_scores.max()
        old_min = X_scores.min()


    # The usual way. Make a function for ease of use
    length = LOF_score.size
    flag = []
    for i in range(0, length):
        if LOF_score[i] > 2:
            # print i
            flag.append(i)

    # Find the outlier percentage
    outlier_percentage = float(len(flag)) / float(len(df))
    print outlier_percentage*100
    percent.append(outlier_percentage)
    if outlier_percentage < .01:
        break

    # Check how many inliers are going to get pruned
    count = 0
    for i in flag:
        if (labels_true[i] != -1):
            count = count + 1

    print('no of inliers pruned is ' + str(count))
    # pruning happens
    df = df.drop(df.index[flag])
    LOF_score = np.delete(LOF_score, flag, 0)
    X_scores = np.delete(X_scores, flag, 0)
    extra_X.extend([X[i] for i in flag])
    X = np.delete(X, flag, 0)
    print(len(labels_true),len(flag))
    outliers_pred.extend([-1] * len(flag))
    true_pred.extend([labels_true[i] for i in flag])
    if counter==0:
        labels_true = np.delete(labels_true.values, flag, 0)
        counter=counter+1
    else:
        labels_true = np.delete(labels_true, flag, 0)

print(outlier_percentage)
# Now get rid of the global outliers
if np.median(percent)!=0.0:
    print("Running KNN")
    knn = KNN(n_neighbors=20, contamination=np.median(percent), method='mean')
    knn.fit(X)
    # get the prediction labels and outlier scores of the training data
    y_pred = knn.labels_
    X_scores = knn.decision_scores_
    LOF_score = X_scores
    # get rid of the outliers
    length = y_pred.size
    flag = []
    for i in range(0, length):
        if y_pred[i] == 1:
            flag.append(i)

    count = 0
    for i in flag:
        if (labels_true[i] != -1):
            count = count + 1

    print('no of inliers pruned is ' + str(count))
    df = df.drop(df.index[flag])

    LOF_score = np.delete(LOF_score, flag, 0)
    X_scores = np.delete(X_scores, flag, 0)
    extra_X.extend([X[i] for i in flag])
    X = np.delete(X, flag, 0)
    # labels_true = np.delete(labels_true.values, flag, 0)
    outliers_pred.extend([-1] * len(flag))
    true_pred.extend([labels_true[i] for i in flag])
    labels_true = np.delete(labels_true, flag, 0)


# print(np.where())

# ax = plt.subplot(gs[0, 1])  # row 0, col 1
# plt.title("After getting rid of outliers by recursively using Local Outlier Factor (LOF)")
# plt.xlim((df['X1'].min() - 100, df['X1'].max() + 100))
# plt.ylim((df['X2'].min() - 100, df['X2'].max() + 100))
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# # plot circles with radius proportional to the outlier scores
# # radius = (X_scores.min() - X_scores) / (X_scores.min() - X_scores.max())
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
hdb = HDBSCAN(min_cluster_size=10, min_samples=5).fit(X)
hdb_labels = hdb.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

print('\n\n++ HDBSCAN Results')
print('Estimated number of clusters: %d' % n_clusters_hdb_)
# print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
# print('Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, hdb_labels))
# print('Completeness: %0.3f' % metrics.completeness_score(labels_true, hdb_labels))
# print('V-measure: %0.3f' % metrics.v_measure_score(labels_true, hdb_labels))
print('Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, hdb_labels))
# # print('Adjusted Mutual Information: %0.3f' % metrics.adjusted_mutual_info_score(labels_true, hdb_labels))
# print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, hdb_labels))
# print('Mislabeled outliers: %0.3f'% np.sum([np.where(hdb_labels[i]!=-1,1,0) for i in np.where(labels_true==-1)]))
# print('DBCV: %0.3f' % DBCV.DBCV(X, hdb_labels))
##############################################################################
# print('Mislabeled points: %0.3f'% (np.sum([np.where(labels_true[i]!=-1,1,0) for i in np.where(hdb_labels==-1)])+np.sum([np.where(hdb_labels[i]!=-1,1,0) for i in np.where(labels_true==-1)])))
# Plot result
import matplotlib.pyplot as plt

# ax = plt.subplot(gs[1, :])
# Black removed and is used for noise instead.
hdb_unique_labels = set(hdb_labels)
# db_unique_labels = set(db_labels)
hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(
    hdb_unique_labels)))  # db_colors = plt.cm.Spectral(np.linspace(0, 1, len(db_unique_labels)))# fig = plt.figure(figsize=plt.figaspect(0.5))# hdb_axis = fig.add_subplot('121')# db_axis = fig.add_subplot('122')
for k, col in zip(hdb_unique_labels, hdb_colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    plt.plot(X[hdb_labels == k, 0], X[hdb_labels == k, 1], 'o', markerfacecolor=col, markeredgecolor='k',
             markersize=6)  # for k, col in zip(db_unique_labels, db_colors):#     if k == -1:#         # Black used for noise.#         col = 'k'#
#     db_axis.lot(X[db_labels == k, 0], X[db_labels == k, 1], 'o', markerfacecolor=col, #markeredgecolor='k', markersize=6)

plt.title('LOF-Recurse-KNN-HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
# db_axis.set_title('DBSCAN\nEstimated number of clusters: %d' % n_clusters_db_)
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

#Code for DBCV output spit-out
# filepath=filepath.replace('Tagged.txt','LOF-Recurse-KNN-HDBSCANPoints.txt')
#
# with open(filepath, 'w') as file:
#     file.writelines('\t'.join(str(j) for j in i) + '\n' for i in X)
#
# filepath=filepath.replace('Points.txt','cluster.txt')
# hdb_labels = [x+1 for x in hdb_labels]
# with open(filepath, 'w') as f:
#     for _list in hdb_labels:
#             #f.seek(0)
#         f.write(str(_list)+'\n')