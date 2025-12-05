import utils
import sys
import cluster_analyzer # Import the new analysis module

def main():
    """
    Performs fast, scalable graph-based clustering on mixed-type data
    and calls an analyzer to interpret the clusters.
    """
    # --- Step 1: Import libraries now that they are installed ---
  
    import pandas as pd
    import numpy as np
    from sklearn.cluster import Birch, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # --- Step 2. Load and Prepare Data (Abalone Dataset) ---
    print("--- Loading Abalone dataset from UCI repository ---")

    # Load the data, assuming the local 'abalone.csv' file has a header row.
    data_for_clustering = pd.read_csv("abalone.csv")

    # For clustering, we'll treat 'Rings' (the age) as a feature, not a target.
    # This is a common practice in unsupervised learning.
    # --- Step 3. Preprocessing for Graph Construction ---
    # UMAP needs numeric input. We use Ordinal Encoding for speed.
    df_processed = data_for_clustering.copy()
    cat_cols = ['Sex']
    for col in cat_cols:
        df_processed[col] = df_processed[col].astype('category').cat.codes

    # --- Step 4. Find Optimal Distance Threshold and Cluster ---
    print("\n--- Finding optimal distance threshold for Agglomerative Clustering ---")
    # We will iterate through different distance thresholds and use the Silhouette Score
    # to evaluate the quality of the resulting clusters. The highest score indicates
    # the best balance between cluster cohesion and separation.

    # Define a range of distance thresholds to test. This range may need tuning.
    thresholds = np.arange(5.0, 25.0, 1.0) 
    best_score = -1
    best_threshold = None
    best_labels = None

    for threshold in thresholds:
        # Configure AgglomerativeClustering with the current threshold
        agg_clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        
        # Run BIRCH with the configured final clusterer
        birch_model = Birch(n_clusters=agg_clusterer)
        birch_model.fit(df_processed)
        labels = birch_model.labels_
        
        # The silhouette score is only defined for 2 or more clusters.
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and n_clusters < len(df_processed):
            score = silhouette_score(df_processed, labels)
            # print(f"Threshold: {threshold:.1f}, Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_labels = labels

    if best_threshold is None:
        print("\nCould not find a suitable clustering. Try adjusting the threshold range in the script.")
        sys.exit(1)

    print(f"\n--- Best Clustering Found ---")
    print(f"Optimal Distance Threshold: {best_threshold}")
    print(f"Resulting Number of Clusters: {len(np.unique(best_labels))}")
    print(f"Best Silhouette Score: {best_score:.3f}\n")
    labels = best_labels

    data_for_clustering['Cluster_ID'] = labels
    print(f"Found {len(np.unique(labels))} clusters.\n")

    # --- Step 5. Interpretation by Statistical Analysis ---
    # This step is now handled by the cluster_analyzer module.
    # It will identify the key features that separate the clusters and profile each one.
    cluster_analyzer.analyze_clusters(data_for_clustering, cat_cols)

if __name__ == "__main__":
    # First, ensure all packages are installed, then run the main analysis.
    utils.install_requirements()
    main()