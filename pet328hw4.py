# We import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

data = pd.read_excel('SPE_shale.xlsx')

# We normalize the numeric data using Standard Scaler from sklearn
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# We determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# We plot the results of the Elbow method to visually determine the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# We perform hierarchical clustering using 'single' or minimum linkage
linked = linkage(scaled_data, 'single')

# We set up labels and getting the corresponding indexes for the dendrogram
labels = data['Lease'].values

# Next, we plotting the dendrogram
plt.figure(figsize=(10, 8))

dendrogram(
    linked,
    orientation='top',
    labels = labels,
    distance_sort='descending',
    show_leaf_counts=True,
    leaf_rotation=90,  
    leaf_font_size=9,
)

plt.title('Hierarchical Clustering Dendrogram (Single linkage)')
plt.xlabel('Wells')
plt.ylabel('Distance')
plt.grid(True)

plt.tight_layout()
plt.show()


# Anomaly Detection
# We use K-Nearest Neighbors for anomaly detection, first with 5 neighbors
nbrs_5 = NearestNeighbors(n_neighbors=5).fit(scaled_data)
distances_5, indices_5 = nbrs_5.kneighbors(scaled_data)
anomaly_score_5 = distances_5.mean(axis=1)

# Then we repeat K-Nearest Neighbors anomaly detection with 10 neighbors
nbrs_10 = NearestNeighbors(n_neighbors=10).fit(scaled_data)
distances_10, indices_10 = nbrs_10.kneighbors(scaled_data)
anomaly_score_10 = distances_10.mean(axis=1)

# We use Local Outlier Factor (LOF) for anomaly detection
lof = LocalOutlierFactor(n_neighbors=5)
lof.fit(scaled_data)
lof_scores = -lof.negative_outlier_factor_  # Negative scores are considered anomalous

# We use Isolation Forest for anomaly detection, first with 1 tree
clf_1 = IsolationForest(n_estimators=1, random_state=0)
clf_1.fit(scaled_data)
scores_1 = -clf_1.decision_function(scaled_data)

# Then we repeat Isolation Forest anomaly detection with 100 trees
clf_100 = IsolationForest(n_estimators=100, random_state=0)
clf_100.fit(scaled_data)
scores_100 = -clf_100.decision_function(scaled_data)


print("Anomaly scores for K-Nearest Neighbors with 5 neighbors:")
print(anomaly_score_5)

print("\nAnomaly scores for K-Nearest Neighbors with 10 neighbors:")
print(anomaly_score_10)

print("\nAnomaly scores for Local Outlier Factor:")
print(lof_scores)

print("\nAnomaly scores for Isolation Forest with 1 tree:")
print(scores_1)

print("\nAnomaly scores for Isolation Forest with 100 trees:")
print(scores_100)


