import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

from load_data import load

def reduce_to_threshold(X: pd.DataFrame, threshold: float = 0.80) -> tuple[pd.DataFrame, int]:
    """
    Scale and reduce X to the number of PCA components needed to explain 'threshold' variance.
    Returns reduced DataFrame and number of components used.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum_var = pca_full.explained_variance_ratio_.cumsum()
    n_components = next(i + 1 for i, val in enumerate(cum_var) if val >= threshold)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    return df_pca, n_components


def main():
    file_path = "data/HR_data_2.csv"
    raw_data, _ = load(file_path)

    biosignal_features = [col for col in raw_data.columns if any(prefix in col for prefix in ['HR_', 'EDA_', 'TEMP_'])]
    questionnaire_features = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                              'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    
    meta_cols = ['Phase', 'Puzzler']

    full_features = biosignal_features + questionnaire_features
    full_data = raw_data[full_features + meta_cols].dropna().reset_index(drop=True)


    X = full_data[full_features]
    metadata = full_data[meta_cols]

    reduced_df, n_components = reduce_to_threshold(X, threshold=0.80)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(reduced_df)
    silhouette_kmeans = silhouette_score(reduced_df, labels_kmeans)

    df_clustered = pd.concat([reduced_df, metadata], axis=1)
    df_clustered["Cluster"] = labels_kmeans

    # PCA scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_clustered, x="PC1", y="PC2", hue="Cluster", style="Puzzler", palette="Set2")
    plt.title(f"PCA (80% variance) with KMeans Clusters (Silhouette: {silhouette_kmeans:.2f})")
    plt.tight_layout()
    plt.show()

    # Questionnaire distribution
    melted = pd.concat([raw_data[questionnaire_features].iloc[X.index], df_clustered["Cluster"]], axis=1).melt(id_vars=["Cluster"])
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted, x="variable", y="value", hue="Cluster")
    plt.xticks(rotation=45)
    plt.title("Questionnaire Responses by Cluster")
    plt.tight_layout()
    plt.show()

    # Cluster vs Phase and Role
    for col, title, cmap in [("Phase", "Phases", "viridis"), ("Puzzler", "Role", "Set2")]:
        pd.crosstab(df_clustered["Cluster"], df_clustered[col], normalize='index').plot(
            kind='bar', stacked=True, colormap=cmap, figsize=(8, 5))
        plt.title(f"Cluster Distribution by {title}")
        plt.xlabel("Cluster")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.show()

    # Mean questionnaire scores by cluster
    pd.concat([raw_data[questionnaire_features].iloc[X.index], df_clustered["Cluster"]], axis=1).groupby("Cluster").mean().T.plot(
        kind='bar', figsize=(14, 6), colormap='Accent')
    plt.title("Average Questionnaire Scores by Cluster")
    plt.xlabel("Questionnaire Item")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ANOVA on questionnaire features
    print("\nANOVA tests for questionnaire features across clusters:")
    for var in questionnaire_features:
        groups = [group[var].dropna().values for name, group in pd.concat([raw_data[questionnaire_features].iloc[X.index], df_clustered[["Cluster"]]], axis=1).groupby("Cluster")]
        stat, p = f_oneway(*groups)
        print(f"{var}: p = {p:.4f}")

if __name__ == "__main__":
    main()
