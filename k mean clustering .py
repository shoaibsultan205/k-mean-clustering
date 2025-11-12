
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\PC\Downloads\life_expectancy.csv")

# Step 2: Select useful columns
X = df[['female','male','both']]

# Step 3: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply K-Means with chosen K (from Elbow, e.g., K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Step 6: Assign meaningful names to clusters
cluster_names = {
    0: 'Low Life Expectancy',
    1: 'Medium Life Expectancy',
    2: 'High Life Expectancy'
}

# Step 7: Visualization
plt.figure(figsize=(12,8))
colors = ['purple', 'green', 'orange']

for cluster in range(3):
    cluster_points = X_scaled[df['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=100, alpha=0.6, color=colors[cluster],
                label=cluster_names[cluster])

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='red', s=300, marker='X', label='Centroids')

plt.title('Countries Clustered by Life Expectancy', fontsize=16)
plt.xlabel('Female Life Expectancy (scaled)', fontsize=14)
plt.ylabel('Male Life Expectancy (scaled)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
