import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time


warnings.filterwarnings(action='ignore')

# read dataset from files
def read_csv(path):
    rawdata = pd.read_csv(path)

    return rawdata
def read_csv_without_header(path):
    rawdata = pd.read_csv(path,header=None)

    return rawdata
# path="d:/1_dow_jones_index.data"
# path="d:/2_Live.csv"
path="d:/3_Sales_Transactions_Dataset_Weekly.csv"
# path="d:/4_water-treatment.data"
range_n_clusters = [2, 3, 4, 5, 6,7,8]

# # test code for path_1
# rawdata=read_csv(path)
# test=rawdata.iloc[:,3:]
# test = test.dropna(axis=0)
# test['open']=test['open'].str.replace(pat='$',repl='',regex=False)
# test['high']=test['high'].str.replace(pat='$',repl='',regex=False)
# test['low']=test['low'].str.replace(pat='$',repl='',regex=False)
# test['close']=test['close'].str.replace(pat='$',repl='',regex=False)
# test['next_weeks_open']=test['next_weeks_open'].str.replace(pat='$',repl='',regex=False)
# test['next_weeks_close']=test['next_weeks_close'].str.replace(pat='$',repl='',regex=False)
#
# print(test)


# test code for path_2
# rawdata=read_csv(path)
# test=rawdata.iloc[:,1:]
# class_le = LabelEncoder()
# test['status_type'] = class_le.fit_transform(rawdata['status_type'].values)
# test = test.dropna(axis=1)
# test = test.drop('status_published',axis='columns')
# print(test)
#


# test code for path_3
rawdata=read_csv(path)
test=rawdata.iloc[:,1:105]
# #


# # test code for path_4
# rawdata=read_csv_without_header(path)
# test=rawdata.iloc[:,1:]
# test=test.replace('?',np.nan)
#
# imr = SimpleImputer(missing_values=np.nan,strategy='mean')
# imp = pd.DataFrame(imr.fit_transform(test))
# imp.columns=test.columns
# imp.index=test.index
# test=imp
#
# print(test)
# #
# data preprocessing
pca = PCA()
fit = pca.fit(test)

features = fit.transform(test)

# summarize components
# print("Explained Variance: %s" % (fit.explained_variance_ratio_))
# print("Singular values: %s" % (fit.singular_values_))
# print(fit.components_)
# print(features)
distortions = []
plt.scatter(features[:,0],features[:,1], c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()
#
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(features)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()
#

# Read   more   about   sklearn   DBSCAN clustering   module
# -https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
mms = MinMaxScaler()
nor_feature=mms.fit_transform(features)
# scaler_ss = StandardScaler().fit(features)
# nor_feature = scaler_ss.transform(features)

# print(nor_feature)
# db = DBSCAN(eps=0.9, min_samples=3, metric='euclidean')
# y_db = db.fit_predict(nor_feature)
# plt.scatter(nor_feature[y_db == 0, 0], nor_feature[y_db == 0, 1],c='lightblue', marker='o',edgecolor='black', label='cluster 1')
# plt.scatter(nor_feature[y_db == 1, 0], nor_feature[y_db == 1, 1],c='red', marker='s', edgecolor='black', label='cluster 2')
# plt.legend()
# plt.tight_layout()
# plt.show()

#
start=time.time()
km = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(features)
#Visualize the clusters identified(using y_km)together with cluster labels.
plt.scatter(features[y_km == 0, 0],features[y_km == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
plt.scatter(features[y_km == 1, 0],features[y_km == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='cluster 2')
plt.scatter(features[y_km == 2, 0],features[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
print ("time : ",time.time()-start)
plt.show()



for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(features)
        # The silhouette_score gives the average value for all the samples.This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(features, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(features, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the y axis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(features[:,0],features[:,1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ""with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()


#Apply agglomerative with 2 clusters
start=time.time()
ac = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
y_ac = ac.fit_predict(features)

plt.scatter(features[y_ac == 0, 0], features[y_ac == 0, 1], c='lightblue',edgecolor='black',marker='o', s=40, label='cluster 1')
plt.scatter(features[y_ac == 1, 0], features[y_ac == 1, 1], c='orange',edgecolor='black',marker='s', s=40, label='cluster 2')
plt.scatter(features[y_ac == 2, 0], features[y_ac == 2, 1], c='lightblue',edgecolor='black',marker='s', s=40, label='cluster 3')
plt.scatter(np.mean(features[y_ac==0,0]),np.mean(features[y_ac==0,1]),s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.scatter(np.mean(features[y_ac==1,0]),np.mean(features[y_ac==1,1]),s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.scatter(np.mean(features[y_ac==2,0]),np.mean(features[y_ac==2,1]),s=250, marker='*',c='red', edgecolor='black',label='centroids')
print ("time : ",time.time()-start)

plt.show()


ag_distortions = []

for i in range(1, 11):
    sse_value=0
    ac = AgglomerativeClustering(n_clusters=i,affinity='euclidean',linkage='complete')
    y_ac_sse = ac.fit_predict(features)
    for k in range(0,i):
        centroid_x=np.mean(features[y_ac_sse==k,0])
        centroid_y=np.mean(features[y_ac_sse==k,1])
        data_x=[]
        data_y=[]
        for j in range(0,len(features[y_ac_sse==k,0])):
            sse_value=sse_value+pow(centroid_x-features[y_ac_sse==k,0][j],2)
            sse_value=sse_value+pow(centroid_y-features[y_ac_sse==k,1][j],2)
    ag_distortions.append(sse_value)
plt.plot(range(1, 11), ag_distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        # cluster_labels = clusterer.fit_predict(features)
        # The silhouette_score gives the average value for all the samples.This gives a perspective into the density and separation of the formed clusters

        ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
        y_ac = ac.fit_predict(features)

        silhouette_avg = silhouette_score(features, y_ac)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(features, y_ac)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[y_ac == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the y axis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(y_ac.astype(float) / n_clusters)
        ax2.scatter(features[:,0],features[:,1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        centroid_x=[]
        centroid_y=[]
        for k in range(0, n_clusters):

            centroid_x.append(np.mean(features[y_ac == k, 0]))
            centroid_y.append(np.mean(features[y_ac == k, 1]))
        # Draw white circles at cluster centers
        ax2.scatter(centroid_x, centroid_y, marker='o', c="white", alpha=1, s=200, edgecolor='k')
        i=0
        for c in range(0,len(centroid_x)):
            ax2.scatter(centroid_x[c], centroid_y[c], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
            i=i+1

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(("Silhouette analysis for Agglomerative clustering on sample data ""with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()