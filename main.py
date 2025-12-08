import utils
import sys
import cluster_analyzer 

def main():
    """
    Performs fast, scalable graph-based clustering on mixed-type data
    and calls an analyzer to interpret the clusters.
    """
    # --- Step 1: Import libraries ---
    import pandas as pd
    import numpy as np
    from sklearn.cluster import Birch, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP
    # Visualization imports
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage # Added for Dendrogram
    import os

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Step 2. Load and Prepare Data ---
    print("--- Loading Abalone dataset from UCI repository ---")
    data_for_clustering = pd.read_csv("data/abalone.csv")

    # --- Step 3. Preprocessing for Graph Construction ---
    # UMAP needs numeric input. Using Ordinal Encoding for speed/simplicity.
    df_processed = data_for_clustering.copy()
    cat_cols = ['Sex']
    for col in cat_cols:
        df_processed[col] = df_processed[col].astype('category').cat.codes
    print("\n-- scale everything ----")
    scaler = StandardScaler()
    df_processed = scaler.fit_transform(df_processed)

    print("\n--- Applying UMAP for dimensionality reduction ---")
    # Configuration optimized for "Feature Extraction" rather than visualization
    umap_model = UMAP(
        n_neighbors=50,      # Higher = Focus on global structure (fewer fragments)
        min_dist=0.0,        # Lower = Tighter, denser clusters for BIRCH
        n_components=10,     # Keep dimensions high enough for clustering info
        n_jobs=3,
        metric='manhattan'   # Often better for mixed/ordinal data
    )
    umap_embedding = umap_model.fit_transform(df_processed)
    print("UMAP embedding created.")

    # --- Step 4. BIRCH Compression (Fit Once) ---
    print("\n--- Compressing data with BIRCH (Running Once) ---")
    # n_clusters=None -> Compresses data to micro-cluster centroids only
    birch_model = Birch(n_clusters=None, threshold=0.3, branching_factor=50)
    birch_model.fit(umap_embedding)
    
    # Extract the compressed centroids (This is what we loop over)
    centroids = birch_model.subcluster_centers_
    print(f"Compressed {len(umap_embedding)} rows into {len(centroids)} micro-clusters.")

    # Map every original data point to its nearest micro-cluster ID
    # We do this ONCE so we don't have to re-predict inside the loop
    micro_cluster_ids = birch_model.predict(umap_embedding)

    # --- OPTIONAL: Visualize Dendrogram ---
    # This helps you visually confirm the natural number of clusters
    print("Generating Dendrogram...")
    plt.figure(figsize=(10, 5))
    # 'ward' linkage minimizes variance, matching Agglomerative defaults
    linked = linkage(centroids, method='ward')
    dendrogram(linked, truncate_mode='lastp', p=30)
    plt.title("Dendrogram (Vertical height = Distance Threshold)")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.savefig('plots/dendrogram.png')
    plt.close()
    print("Dendrogram saved to plots/dendrogram.png")

    # --- Step 5. Loop: Find Optimal Clusters on Centroids ---
    print("\n--- Finding optimal distance threshold (Auto-detecting K) ---")
    
    best_score = -1
    best_model = None
    best_labels = None
    best_n_clusters = 0
    
    # Range of thresholds to try. 
    # Since UMAP distances can vary, we scan a wide range.
    # We look at the linkage matrix to guess reasonable bounds, or just scan broad:
    # (Start low, go high. Higher threshold = Fewer clusters)
    thresholds_to_try = np.linspace(0.5, 50, 50) 
    
    valid_clustering_found = False

    for thresh in thresholds_to_try:
        # 1. Cluster Centroids using Threshold
        # n_clusters=None is REQUIRED when using distance_threshold
        agg_clusterer = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=thresh,
            linkage='ward' 
        )
        
        centroid_labels = agg_clusterer.fit_predict(centroids)
        n_clusters_found = len(set(centroid_labels))
        
        # 2. Safety Filter: Ignore if cluster count is crazy
        # We only care if the algorithm finds between 3 and 15 clusters.
        # If it finds 30 (too granular) or 1 (too broad), skip it.
        if n_clusters_found < 2 or n_clusters_found > 15:
            continue
            
        valid_clustering_found = True
        
        # 3. Map back to full data for scoring
        full_labels = centroid_labels[micro_cluster_ids]
        
        # 4. Calculate Score
        score = silhouette_score(umap_embedding, full_labels)
        print(f"Threshold={thresh:.1f} -> Found {n_clusters_found} clusters. Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters_found
            best_labels = full_labels
            best_model = agg_clusterer

    if not valid_clustering_found:
        print("\nNo threshold produced a valid cluster count (2-15).")
        print("Try increasing the max value in 'thresholds_to_try'.")
        sys.exit(1)

    print(f"\n--- Best Clustering Found ---")
    print(f"Optimal Threshold: {best_model.distance_threshold:.2f}")
    print(f"Resulting Clusters: {best_n_clusters}")
    print(f"Best Silhouette Score: {best_score:.3f}\n")
    
    # Assign final labels
    data_for_clustering['Cluster_ID'] = best_labels
    labels = best_labels

    # --- Step 6: Visualize the clusters with t-SNE ---
    # (Note: You could also plot the UMAP embedding directly since you already have it,
    # but t-SNE is sometimes preferred for final viz. Keeping your original logic here.)
    print("--- Applying t-SNE for visualization ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    # Using df_processed (original features) for t-SNE to see if clusters hold up 
    # in original space, not just UMAP space.
    tsne_embedding = tsne.fit_transform(df_processed)
    print("t-SNE embedding created.")

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_points = tsne_embedding[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors(i), label=f'Cluster {label}', s=15, alpha=0.7)

    plt.title(f'2D t-SNE Projection (k={best_n_clusters})', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plot_path = 'plots/tsne_cluster_visualization.png'
    plt.savefig(plot_path)
    print(f"Cluster visualization saved to: {plot_path}\n")
    plt.close()

    # --- Step 7. Interpretation ---
    cluster_analyzer.analyze_clusters(data_for_clustering, cat_cols)

if __name__ == "__main__":
    utils.install_requirements()
    main()