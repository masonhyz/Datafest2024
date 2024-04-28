import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


cluster_to_category = {
    0: 'Category A',
    1: 'Category B',
    2: 'Category C',
    3: 'Category D',
    4: 'Category E'
}

# FULL = True
# if FULL:
#     dir = 'full_03_04'
# else:
#     dir = "Random Sample of Data Files_03_04"

# # checkpoint_eoc = pd.read_csv(os.path.join(dir, "checkpoints_eoc.csv"))
# # checkpoint_pulse = pd.read_csv(os.path.join(dir, "checkpoints_pulse.csv"))
# # items = pd.read_csv(os.path.join(dir, "items.csv"))
# # media_views = pd.read_csv(os.path.join(dir, "media_views.csv"))
# # page_views = pd.read_csv(os.path.join(dir, "page_views.csv"))
# responses = pd.read_csv(os.path.join(dir, "responses.csv"))
# print("data loaded")

# responses['correctness'] = responses['points_earned'] / responses['points_possible']
# responses['type'] = responses.apply(lambda row: row['item_type'] if row['item_type'] == 'code' else row['lrn_type'], axis=1)
# _responses = responses[['type', 'student_id', 'correctness']]
# _correctness_matrix = _responses.pivot_table(index='student_id', columns='type', values='correctness', fill_value=0)






# _correctness_matrix.to_csv('_correctness_matrix.csv', index=False)
_correctness_matrix = pd.read_csv("_correctness_matrix.csv")
print('data loaded')
# _correctness_matrix = _correctness_matrix[[column for column in _correctness_matrix.columns if column not in ['sortlist', 'formulaV2', 'plaintext']]]






# max_k = 20
# sse = []
# for k in range(1, max_k):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(_correctness_matrix)
#     sse.append(kmeans.inertia_)
# plt.plot(range(1, max_k), sse, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.show()

# kmeans = KMeans(n_clusters=5, random_state=1110)
# clusters = kmeans.fit_predict(_correctness_matrix)

# _correctness_matrix['cluster'] = clusters
# cluster_means = _correctness_matrix.groupby('cluster').mean()
# print(cluster_means)





# fitting PCA 2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(_correctness_matrix)
# fitting GMM on reconstruction from PCA
reduced_data1 = pd.DataFrame(reduced_data)
gmm = GaussianMixture(n_components=5, covariance_type='tied', random_state=2)
gmm.fit(reduced_data1)
clusters = gmm.predict(reduced_data1)
reduced_data1['cluster'] = clusters
reduced_data1['cluster'] = reduced_data1['cluster'].map(cluster_to_category)

plt.figure(figsize=(8, 6), dpi=300)  # High resolution for publication quality
# Create a scatter plot
scatter = plt.scatter(reduced_data1.iloc[:, 0], reduced_data1.iloc[:, 1],
                      c=clusters, cmap='viridis', marker='o',
                    #   edgecolor='k',  # Black edge color for markers
                      s=50,  # Size of markers
                      alpha=0.8)  # Slight transparency for better visibility of overlapping points
plt.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
# Title and labels with a larger font size
plt.title('Clusters of GMM on the reconstruction from PCA', fontsize=18)
plt.xlabel('Principal Component 1', fontsize=16)
plt.ylabel('Principal Component 2', fontsize=16)
# Larger tick marks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Color bar with a title, adjusted using the returned value from scatter
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Label', fontsize=16)
cbar.ax.tick_params(labelsize=14) 
plt.tight_layout()
plt.savefig('2d cluster.png', format='png')  # Save as a .png file with high resolution
plt.show()



# 3D PCA and EM with GMM
pca = PCA(n_components=3)
reduced_data_3d = pca.fit_transform(_correctness_matrix)

reduced_data2 = pd.DataFrame(reduced_data_3d)
gmm = GaussianMixture(n_components=5, covariance_type='tied', random_state=2)
gmm.fit(reduced_data2)
clusters = gmm.predict(reduced_data2)
reduced_data2['cluster'] = clusters
reduced_data2['cluster'] = reduced_data2['cluster'].map(cluster_to_category)


fig = px.scatter_3d(reduced_data2, x=0, y=1, z=2,
                    color=clusters,
                    title='3D PCA Visualization of GMM student clusters',
                    labels={'Cluster': 'Cluster'},
                    opacity=0.7,  # Adjust opacity to ensure plot is not overcrowded visually,
                    color_continuous_scale='viridis'
                    )  
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=30),  # Reduce margin to make it more compact
    paper_bgcolor='white',  # Background color for the export
    scene=dict(
        xaxis=dict(title='PC1', backgroundcolor="rgba(0, 0, 0, 0)"),
        yaxis=dict(title='PC2', backgroundcolor="rgba(0, 0, 0, 0)"),
        zaxis=dict(title='PC3', backgroundcolor="rgba(0, 0, 0, 0)"),
        xaxis_showgrid=True, xaxis_gridwidth=1, xaxis_gridcolor='lightgrey',
        yaxis_showgrid=True, yaxis_gridwidth=1, yaxis_gridcolor='lightgrey',
        zaxis_showgrid=True, zaxis_gridwidth=1, zaxis_gridcolor='lightgrey'
    ),
    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera view angle
    )
)
fig.show()



pcs = pca.components_
print(pcs)
# To see the explained variance ratio of each principal component:
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
# Getting principal components w.r.t. our features
features = _correctness_matrix.columns
print(features)
for i, component in enumerate(pcs):
    component_loadings = dict(zip(features, component))
    sorted_loadings = sorted(component_loadings.items(), key=lambda x: x[1], reverse=True)
    print(f"Principal Component {i+1}:")
    for feature, loading in sorted_loadings:
        print(f"{feature}: {loading:.3f}")
    print("")


# Printing feature means of each cluster
_correctness_matrix['cluster'] = clusters
cluster_means = _correctness_matrix.groupby('cluster').mean()
print(cluster_means)
