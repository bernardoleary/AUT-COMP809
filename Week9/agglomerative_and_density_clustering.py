from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            edgecolor='black',
            label='cluster 1'
            )

plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='cluster 2')

plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            edgecolor='black',
            label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=150,
            marker='*',
            c='red',
            label='Centroids')

plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('P:/COMP809/11_05.png', dpi=300)
plt.show()

"""
# Agglomerative — Bottom up approach.
# Start with many small clusters and merge them together to create bigger clusters.

# Organizing clusters as a hierarchical tree
# Clustering data using Agglomerative Clustering in bottom-upfashion
# For more details -
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

1. Initially, all the data-points are a cluster of its own.
2. Take two nearest clusters and join them to form one single cluster.
3. Proceed recursively step 2 until you obtain the desired number of clusters.

"""

ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
labels_1 = ac.fit_predict(X)
print('Cluster labels Agglomerative with 3 clusters: %s' % labels_1)
# The output is a one-dimensional array
# of 150 elements corresponding to the clusters assigned to our 150 data points.

plt.scatter(X[labels_1 == 0, 0],
            X[labels_1 == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            edgecolor='black',
            label='cluster 1')

plt.scatter(X[labels_1 == 1, 0],
            X[labels_1 == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='cluster 2')

plt.scatter(X[labels_1 == 2, 0],
            X[labels_1 == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            edgecolor='black',
            label='cluster 3')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

"""
Do not have to assume any particular number
of clusters
  Any desired number of clusters can be obtained
  
by ‘cutting’ the dendogram at the proper level
  They may correspond to meaningful taxonomies
  
  Example in biological sciences (e.g., animal， kingdom, phylogeny reconstruction, …)
"""

# increase the level of abstraction by decreasing the number of clusters
ac_2 = AgglomerativeClustering(n_clusters=2,
                               affinity='euclidean',
                               linkage='complete')

labels_2 = ac_2.fit_predict(X)
print('Cluster labels Agglomrtative with 2 clusters: %s' % labels_2)
# The output is a one-dimensional array
# of 150 elements corresponding to the clusters assigned to our 150 data points.


plt.scatter(X[labels_2 == 0, 0],
            X[labels_2 == 0, 1],
            c='lightgreen',
            marker='s',
            edgecolor='black',
            label='cluster 1')

plt.scatter(X[labels_2 == 1, 0],
            X[labels_2 == 1, 1],
            c='orange',
            edgecolor='black',
            marker='o',
            label='cluster 2')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ------- ------- ------- ------- ------- ------- ------- ------- ------- -------
# n_clusters – in hierarchical clustering is specify a cut-of point
# There is no definitive num of clusters since cluster analysis is essentially an exploratory approach;
# the interpretation of the resulting hierarchical structure is context-dependent
# and often several solutions are equally good from a theoretical point of view.
#
# There is no definitive answer since
# cluster analysis is essentially an exploratory approach;
# the interpretation of the resulting hierarchical structure is context-dependent
# and often several solutions are equally good from a theoretical point of view.
# domain-specific knowledge is required to analyze whether the result makes sense or not.

# Children of hierarchical clustering
children = ac_2.children_

# Distances between each pair of children
# Since we don't have this information, we can use a uniform one for plotting
distance = np.arange(children.shape[0])

# The number of observations contained in each cluster level
no_of_observations = np.arange(2, children.shape[0] + 2)

# Create linkage matrix and then plot the dendrogram
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

# Plot the corresponding dendrogram
dendrogram(linkage_matrix, labels=ac_2.labels_, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
# https://towardsdatascience.com/hierarchical-clustering-explained-e59b13846da8
# ------- ------- ------- ------- ------- ------- ------- ------- ------- -------

"""

Why do we need to learn different clustering algorithm??

First, let’s clear up the role of clustering.

Clustering is an unsupervised learning technique where we try to group the data points based on specific characteristics.
 There are various clustering algorithms with K-Means and Hierarchical being the most used ones. 
 Some of the use cases of clustering algorithms include:
         Document Clustering
         Recommendation Engine
         Image Segmentation
         Market Segmentation
         Search Result Grouping
         and Anomaly Detection
         and etc.

K-Means and Hierarchical Clustering both fail in creating clusters of more arbitrary shapes 
(like non-spherical (non-globular) shapes). 
They are not able to form clusters based on varying densities. 

Let’s try to understand it with an example. Here we have data points densely present in the form of concentric circles:

____________________________________________________________________________________________
____________________________________________________________________________________________


# Now we are going to create two interleaving half circles using sklearn make_moons function.
# Read more on-
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

make_circles and make_moons generate 2d binary classification datasets 
that are challenging to certain algorithms 
(e.g. centroid-based clustering or linear classification),
 including optional Gaussian noise. 
 They are useful for visualisation. 
 make_circles produces Gaussian data 
 with a spherical decision boundary 
 for binary classification, 
 while make_moons produces two interleaving half circles.
"""

# make a random moon dataset
X_moon, y_moon = make_moons(n_samples=100, noise=0.05, random_state=0)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon)
plt.tight_layout()
plt.show()

# Compare K-means and hierarchical clustering clustering:
# Configure the plot

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Apply k-means with 2 clusters

km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X_moon)
ax1.scatter(X_moon[y_km == 0, 0],
            X_moon[y_km == 0, 1],
            edgecolor='black',
            c='lightblue',
            marker='o',
            s=50,
            label='cluster 1')

ax1.scatter(X_moon[y_km == 1, 0],
            X_moon[y_km == 1, 1],
            edgecolor='black',
            c='red',
            marker='s',
            s=50,
            label='cluster 2')

ax1.set_title('K-means clustering')

# Apply agglomerative with 2 clusters

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')

y_ac = ac.fit_predict(X_moon)

ax2.scatter(X_moon[y_ac == 0, 0],
            X_moon[y_ac == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=50,
            label='cluster 1')

ax2.scatter(X_moon[y_ac == 1, 0],
            X_moon[y_ac == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=50,
            label='cluster 2')

ax2.set_title('Agglomerative Clustering')
plt.legend()
plt.show()

"""

K-Means and 
Hierarchical Clustering both FAIL in creating clusters of more arbitrary shapes. 
They are not able to form clusters based on varying densities. 
That’s why we need DBSCAN clustering.

# Now apply the DBSCAN clusterer with min_samples =4 and the eps
parameter = 0.3.
Choosing optimal values for these two parameters are critical
and using unsuitable values may highly effect to the performance
of the algorithm/ results.
In this particular case eps = 0.5 (default value) failed to
detect two distinguished clusters.
#Read more about sklearn DBSCAN clustering module -
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

 Basic algorithm is straightforward
1. Compute the proximity matrix
2. Let each data point be a cluster
3. Repeat
4. Merge the two closest clusters
5. Update the proximity matrix
6. Until only a single cluster remains

 Key operation is the computation of the proximity of two clusters
 Different approaches to defining the distance between clusters distinguish the different algorithms

"""

db = DBSCAN(eps=0.3, min_samples=4, metric='euclidean')
y_db = db.fit_predict(X_moon)
plt.scatter(X_moon[y_db == 0, 0],
            X_moon[y_db == 0, 1],
            c='lightblue',
            marker='o',
            s=50,
            edgecolor='black',
            label='cluster 1')

plt.scatter(X_moon[y_db == 1, 0],
            X_moon[y_db == 1, 1],
            c='red',
            marker='s',
            s=50,
            edgecolor='black',
            label='cluster 2')
plt.legend()
plt.tight_layout()
plt.show()

"""
K-means

tries to find cluster centers that are representative of certain regions of the data
alternates between two steps: assigning each data point to the closest cluster center, 
and then setting each cluster center as the mean of the data points that are assigned to it
the algorithm is finished when the assignment of instances to clusters no longer changes
"""

"""
DBSCAN
stands for “density based spatial clustering of applications with noise”

does NOT require the user to set the number of clusters a priori

can capture clusters of complex shapes!

can identify points that are NOT part of any cluster (very useful as outliers detector!)

is somewhat slower than agglomerative clustering and k-means, 
but still scales to relatively large datasets.
works by identifying points that are in crowded regions of the feature space,
 where many data points are close together (dense regions in feature space)
Points that are within a dense region are called core samples (or core points)

There are two parameters in DBSCAN: min_samples and eps
If there are at least min_samples many data points within a distance of eps to a given data point,
 that data point is classified as a core sample
 
core samples that are closer to each other than 
the distance eps are put into the same cluster by DBSCAN.
"""
