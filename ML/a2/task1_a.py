import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings(action='ignore')

# read dataset from files
def read_csv(path):
    rawdata = pd.read_csv(path)

    return rawdata

# path=["d:/1_dow_jones_index.data" ]
path="d:/2_Live.csv"

range_n_clusters = [2, 3, 4, 5, 6,7,8]

# test code
# rawdata=read_csv(path)
# test=rawdata.iloc[:,3:11]
# # print(test)
#
#
#
# pca = PCA()
# fit = pca.fit(test)
# features = fit.transform(test)
## summarize components
# print("Explained Variance: %s" % (fit.explained_variance_ratio_))
# print("Singular values: %s" % (fit.singular_values_))
# print(fit.components_)
# print(features)
# distortions = []
# plt.scatter(features[:,0],features[:,1], c='white', marker='o', edgecolor='black', s=50)
# plt.grid()
# plt.tight_layout()
# plt.show()
#
# for i in range(1, 11):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     km.fit(features)
#     distortions.append(km.inertia_)
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.tight_layout()
# plt.show()
#
# km = KMeans(n_clusters=5,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
# y_km = km.fit_predict(features)
# #Visualize the clusters identified(using y_km)together with cluster labels.
# plt.scatter(features[y_km == 0, 0],features[y_km == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
# plt.scatter(features[y_km == 1, 0],features[y_km == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='cluster 2')
# plt.scatter(features[y_km == 2, 0],features[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')
# plt.scatter(features[y_km == 3, 0],features[y_km == 3, 1],s=50, c='red',marker='v', edgecolor='black',label='cluster 4')
# plt.scatter(features[y_km == 4, 0],features[y_km == 4, 1],s=50, c='green',marker='v', edgecolor='black',label='cluster 5')
# plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.tight_layout()
#
# plt.show()
#test code
path="d:/3_Sales_Transactions_Dataset_Weekly.csv"


range_n_clusters = [2, 3, 4, 5, 6,7,8]

#test code
rawdata=read_csv(path)
test=rawdata.iloc[:,1:105]
# print(test)

pca = PCA(n_components=2)
fit = pca.fit(test)
features = fit.transform(test)
## summarize components
print("Explained Variance: %s" % (fit.explained_variance_ratio_))
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
km = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(features)
#Visualize the clusters identified(using y_km)together with cluster labels.
plt.scatter(features[y_km == 0, 0],features[y_km == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
plt.scatter(features[y_km == 1, 0],features[y_km == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='cluster 2')
plt.scatter(features[y_km == 2, 0],features[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
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

# # test code
# path="d:/4_water-treatment.data"
#
#
# range_n_clusters = [2, 3, 4, 5, 6,7,8]
#
# #test code
# rawdata=read_csv(path)
# test=rawdata.iloc[:,1:]
# # print(test)
#
# pca = PCA(n_components=5)
# fit = pca.fit(test)
# features = fit.transform(test)
# ## summarize components
# print("Explained Variance: %s" % (fit.explained_variance_ratio_))
# print("Singular values: %s" % (fit.singular_values_))
# print(fit.components_)
# print(features)
#
# distortions = []
# plt.scatter(features[:,0],features[:,1], c='white', marker='o', edgecolor='black', s=50)
# plt.grid()
# plt.tight_layout()
# plt.show()
# #
# for i in range(1, 11):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     km.fit(features)
#     distortions.append(km.inertia_)
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.tight_layout()
# plt.show()
# #
# km = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
# y_km = km.fit_predict(features)
# #Visualize the clusters identified(using y_km)together with cluster labels.
# plt.scatter(features[y_km == 0, 0],features[y_km == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
# plt.scatter(features[y_km == 1, 0],features[y_km == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='cluster 2')
# plt.scatter(features[y_km == 2, 0],features[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')
#
# plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.tight_layout()
#
# plt.show()
# test code


#
# # repeat the process with each file
# for path_value in path:
#     rawdata=read_csv(path_value)
#     test=rawdata[:]
#     # print(test.columns)
#     X=test.iloc[:,8]
#     Y=test.iloc[:,15]
#     K=pd.merge(X,Y,how="outer",left_index=True,right_index=True)
#     # print(K)
#     # plt.scatter(X,Y, c='white', marker='o', edgecolor='black', s=50)
#     # plt.grid()
#     # plt.tight_layout()
#     # plt.show()
#     distortions = []
#
#     for i in range(1, 11):
#         km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#         km.fit(K)
#         distortions.append(km.inertia_)
#     plt.plot(range(1, 11), distortions, marker='o')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Distortion')
#     plt.tight_layout()
#     plt.show()
#
#     for n_clusters in range_n_clusters:
#         # Create a subplot with 1 row and 2 columns
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         fig.set_size_inches(18, 7)
#
#         # The 1st subplot is the silhouette plot
#         # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
#         ax1.set_xlim([-0.1, 1])
#
#         # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
#         ax1.set_ylim([0, len(K) + (n_clusters + 1) * 10])
#
#         # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
#         clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#         cluster_labels = clusterer.fit_predict(K)
#         # The silhouette_score gives the average value for all the samples.This gives a perspective into the density and separation of the formed clusters
#         silhouette_avg = silhouette_score(K, cluster_labels)
#         print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
#
#         # Compute the silhouette scores for each sample
#         sample_silhouette_values = silhouette_samples(K, cluster_labels)
#         y_lower = 10
#         for i in range(n_clusters):
#             ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#             ith_cluster_silhouette_values.sort()
#             size_cluster_i = ith_cluster_silhouette_values.shape[0]
#             y_upper = y_lower + size_cluster_i
#             color = cm.nipy_spectral(float(i) / n_clusters)
#             ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
#                               edgecolor=color, alpha=0.7)
#             ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#             y_lower = y_upper + 10
#         ax1.set_title("The silhouette plot for the various clusters.")
#         ax1.set_xlabel("The silhouette coefficient values")
#         ax1.set_ylabel("Cluster label")
#
#         ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#         ax1.set_yticks([])  # Clear the y axis labels / ticks
#         ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#         colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#         ax2.scatter(X, Y, marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
#         centers = clusterer.cluster_centers_
#
#         # Draw white circles at cluster centers
#         ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
#         for i, c in enumerate(centers):
#             ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
#
#         ax2.set_title("The visualization of the clustered data.")
#         ax2.set_xlabel("Feature space for the 1st feature")
#         ax2.set_ylabel("Feature space for the 2nd feature")
#         plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ""with n_clusters = %d" % n_clusters),
#                      fontsize=14, fontweight='bold')
#     plt.show()