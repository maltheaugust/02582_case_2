import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from load_data import load




def reduce_to_threshold(raw_data: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    raw_data_clean = raw_data.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_data_clean)
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    cum_var = pca_full.explained_variance_ratio_.cumsum()
    n_components = next(i + 1 for i, val in enumerate(cum_var) if val >= threshold)
    pca_threshold = PCA(n_components=n_components)
    X_pca_thresh = pca_threshold.fit_transform(X_scaled)
    return pd.DataFrame(X_pca_thresh)







def main():
    # load dataset 
    file_path = "data/HR_data_2.csv"
    raw_data, metadata = load(file_path)

    min_variance_explained = 0.8

    reduced_df = reduce_to_threshold(raw_data, threshold=min_variance_explained)

    # biosignal, questionnaire, and puzzler features
    biosignal_features = [col for col in raw_data.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    questionnaire_features = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                            'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']

    puzzler_feature = "Puzzler"
    cluster_col = "Cluster"
    phase_col = "Phase"
    round_col = "Round"


    # PCA for visualization
    pca = PCA(n_components=10) # ~80 variance explained with 10 components
    X_pca = pca.fit_transform(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)

    # Combine data for analysis
    df_clustered = pd.concat([raw_data_clean, metadata_clean], axis=1)
    df_clustered["Cluster"] = labels_kmeans
    df_clustered["PCA1"] = X_pca[:, 0]
    df_clustered["PCA2"] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_clustered, x="PCA1", y="PCA2", hue="Cluster", style="Puzzler", palette="Set2")
    plt.title(f"PCA of Biosignals with KMeans Clusters (Silhouette: {silhouette_kmeans:.2f})")
    plt.tight_layout()
    plt.show()

    # Boxplot of questionnaire responses by cluster
    melted = df_clustered.melt(id_vars=["Cluster"], value_vars=questionnaire_features)
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted, x="variable", y="value", hue="Cluster")
    plt.xticks(rotation=45)
    plt.title("Questionnaire Scores by Cluster (KMeans)")
    plt.tight_layout()
    plt.show()

    # Cross-tab analyses
    for col, title, cmap in [("Phase", "Phases", "viridis"), ("Round", "Rounds", "plasma"), ("Puzzler", "Role", "Set2")]:
        table = pd.crosstab(df_clustered["Cluster"], df_clustered[col], normalize='index')
        if col == "Puzzler":
            table.columns = ["Instructor", "Puzzler"]
        table.plot(kind='bar', stacked=True, colormap=cmap, figsize=(8, 5))
        plt.title(f"Cluster Distribution Across {title}")
        plt.xlabel("Cluster")
        plt.ylabel("Proportion")
        plt.legend(title=title)
        plt.tight_layout()
        plt.show()


    # Average questionnaire scores
    df_clustered.groupby("Cluster")[questionnaire_features].mean().T.plot(kind='bar', figsize=(14, 6), colormap='Accent')
    plt.title("Average Questionnaire Scores by Cluster")
    plt.xlabel("Questionnaire Item")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # GMM Clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    silhouette_gmm = silhouette_score(X_scaled, gmm_labels)

    df_gmm = pd.concat([raw_data_clean, metadata_clean], axis=1)
    df_gmm["Cluster"] = gmm_labels
    df_gmm["PCA1"] = X_pca[:, 0]
    df_gmm["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_gmm, x="PCA1", y="PCA2", hue="Cluster", style="Puzzler", palette="Set2")
    plt.title(f"PCA of Biosignals with GMM Clusters (Silhouette: {silhouette_gmm:.2f})")
    plt.tight_layout()
    plt.show()


    # ANOVA tests on questionnaire features
    print("\nANOVA tests for questionnaire features across KMeans clusters:")
    for var in questionnaire_features:
        groups = [group[var].dropna().values for name, group in df_clustered.groupby("Cluster")]
        stat, p = f_oneway(*groups)
        print(f"{var}: p = {p:.4f}")


# # Prepare physiological data for clustering
# X = df[biosignal_features].dropna()
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # PCA for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # KMeans clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# labels = kmeans.fit_predict(X_scaled)

# # Add clustering results back to original dataframe
# df_clustered = df.loc[X.index].copy()
# df_clustered["Cluster"] = labels
# df_clustered["PCA1"] = X_pca[:, 0]
# df_clustered["PCA2"] = X_pca[:, 1]

# # Silhouette score
# silhouette = silhouette_score(X_scaled, labels)

# # Plot PCA results with cluster coloring
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_clustered, x="PCA1", y="PCA2", hue="Cluster", style=puzzler_feature, palette="Set2")
# plt.title(f"PCA of Physiological Features with KMeans Clusters (Silhouette: {silhouette:.2f})")
# plt.tight_layout()
# plt.show()

# # Questionnaire score distributions by cluster
# melted = df_clustered.melt(id_vars=["Cluster"], value_vars=questionnaire_features)
# plt.figure(figsize=(14, 6))
# sns.boxplot(data=melted, x="variable", y="value", hue="Cluster")
# plt.xticks(rotation=45)
# plt.title("Distribution of Questionnaire Responses by Cluster")
# plt.tight_layout()
# plt.show()



# # Summary of clusters by questionnaire means
# summary = df_clustered.groupby("Cluster")[questionnaire_features + [puzzler_feature]].mean()
# print(summary)


# # Count of puzzler/instructor per cluster
# role_dist = pd.crosstab(df_clustered['Cluster'], df_clustered[puzzler_feature])
# print(role_dist)
# role_dist.plot(kind="bar", stacked=True, colormap="Pastel1")
# plt.title("Role Distribution Across Clusters")
# plt.xlabel("Cluster")
# plt.ylabel("Count")
# plt.legend(["Instructor", "Puzzler"], title="Role")
# plt.tight_layout()
# plt.show()


from scipy.stats import f_oneway, kruskal

for var in questionnaire_features:
    groups = [group[var].dropna().values for name, group in df_clustered.groupby('Cluster')]
    stat, p = f_oneway(*groups)
    print(f"{var}: ANOVA p-value = {p:.4f}")



# # Drop rows with missing values in either set
# df_clean = df[biosignal_features + questionnaire_features].dropna()


# # === OPTION 1: Clustering on Biosignals Only ===
# X_bio = df_clean[biosignal_features]
# X_bio_scaled = StandardScaler().fit_transform(X_bio)

# # PCA for visualization
# pca_bio = PCA(n_components=2)
# X_bio_pca = pca_bio.fit_transform(X_bio_scaled)

# # KMeans
# kmeans_bio = KMeans(n_clusters=3, random_state=42)
# labels_bio = kmeans_bio.fit_predict(X_bio_scaled)
# silhouette_bio = silhouette_score(X_bio_scaled, labels_bio)

# # Visualization
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_bio_pca[:, 0], y=X_bio_pca[:, 1], hue=labels_bio, palette='Set2')
# plt.title(f"Biosignal Clustering (Silhouette: {silhouette_bio:.2f})")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend(title="Cluster")
# plt.tight_layout()
# plt.show()

# # === OPTION 2: Clustering on Questionnaire Data Only ===
# X_q = df_clean[questionnaire_features]
# X_q_scaled = StandardScaler().fit_transform(X_q)

# # PCA for visualization
# pca_q = PCA(n_components=2)
# X_q_pca = pca_q.fit_transform(X_q_scaled)

# # KMeans clustering
# kmeans_q = KMeans(n_clusters=3, random_state=42)
# labels_q = kmeans_q.fit_predict(X_q_scaled)
# silhouette_q = silhouette_score(X_q_scaled, labels_q)

# # Visualization
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_q_pca[:, 0], y=X_q_pca[:, 1], hue=labels_q, palette='Set1')
# plt.title(f"Questionnaire Clustering (Silhouette: {silhouette_q:.2f})")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend(title="Cluster")
# plt.tight_layout()
# plt.show()
