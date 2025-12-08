import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def generate_summary_sentence(cluster_id, size, characteristics):
    """
    Generates a natural language summary for a single cluster based on its key features.

    Args:
        cluster_id (int): The ID of the cluster.
        size (int): The number of members in the cluster.
        characteristics (list): A list of tuples, where each tuple describes a key feature.
                                e.g., [('Rings', 'High'), ('Sex', "Dominated by 'M'")]

    Returns:
        str: A descriptive sentence summarizing the cluster.
    """
    if not characteristics:
        return f"Cluster {cluster_id} (Size: {size}) has no strong differentiating features among the top characteristics."

    descriptions = []
    for feature, label in characteristics:
        if label == 'High':
            descriptions.append(f"high '{feature}'")
        elif label == 'Low':
            descriptions.append(f"low '{feature}'")
        elif label.startswith("Dominated by"):
            category = label.split("'")[1]
            descriptions.append(f"a predominance of '{category}' individuals")

    # Join the descriptions into a natural-sounding list
    desc_str = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}" if len(descriptions) > 1 else descriptions[0]

    return f"Cluster {cluster_id} (Size: {size}) is primarily characterized by {desc_str}."

def analyze_clusters(df, cat_cols):
    """
    Analyzes clusters to find key differentiating features using ANOVA and Chi-Square tests.
    First, it ranks all features by their statistical power in separating clusters.
    Then, it profiles each cluster against the top 5 features.

    Args:
        df (pd.DataFrame): DataFrame containing the original data plus a 'Cluster_ID' column.
        cat_cols (list): List of column names that are categorical.
    """
    print("--- Starting Cluster Analysis ---")
    
    if 'Cluster_ID' not in df.columns:
        print("Error: 'Cluster_ID' column not found in DataFrame.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.drop('Cluster_ID')
    
    feature_scores = {}

    # 1. Continuous Features: One-way ANOVA to get F-statistic
    print("Running ANOVA for continuous features...")
    for feature in numeric_cols:
        # Create a list of arrays, where each array contains the feature's values for a cluster
        groups = [group[feature].values for name, group in df.groupby('Cluster_ID')]
        
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            if not np.isnan(f_stat):
                feature_scores[feature] = f_stat

    # 2. Categorical Features: Chi-Square Test
    print("Running Chi-Square test for categorical features...")
    for feature in cat_cols:
        contingency_table = pd.crosstab(df[feature], df['Cluster_ID'])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        feature_scores[feature] = chi2_stat

    # Get top 5 differentiating features based on their scores
    if not feature_scores:
        print("Could not calculate feature importance scores.")
        return
        
    top_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    top_feature_names = [f[0] for f in top_features]
    
    print(f"\nTop 5 differentiating features: {top_feature_names}")

    # --- Generate natural language summaries for each cluster ---
    global_means = df[numeric_cols].mean()
    cluster_ids = sorted(df['Cluster_ID'].unique())

    print("\n--- Cluster Summaries ---")
    for cluster_id in cluster_ids:
        cluster_data = df[df['Cluster_ID'] == cluster_id]
        other_data = df[df['Cluster_ID'] != cluster_id]
        
        # Gather characteristics for the summary sentence
        characteristics = []
        for feature in top_feature_names:
            if feature in numeric_cols:
                # Reverting to a simple comparison against the global mean for the summary.
                cluster_mean = cluster_data[feature].mean()
                if cluster_mean > global_means[feature]:
                    label = "High"
                else:
                    label = "Low"
                characteristics.append((feature, label))

            elif feature in cat_cols:
                # Find the most over-represented category in the cluster using Lift
                cluster_dist = cluster_data[feature].value_counts(normalize=True)
                global_dist = df[feature].value_counts(normalize=True)
                
                lift_scores = (cluster_dist / global_dist).fillna(0)
                
                if not lift_scores.empty:
                    most_defining_category = lift_scores.idxmax()
                    characteristics.append((feature, f"Dominated by '{most_defining_category}'"))

        # Generate and print the final summary sentence for the cluster
        summary = generate_summary_sentence(cluster_id, len(cluster_data), characteristics)
        print(summary)

    # --- Generate and Export Cluster Comparison Data and Visuals ---
    print("\n--- Generating and exporting comparison visuals and data ---")
    summary_parts = []
    top_numeric_features = [f for f in top_feature_names if f in numeric_cols]
    if top_numeric_features:
        numeric_summary_df = pd.DataFrame(global_means[top_numeric_features], columns=['Global'])
        for cluster_id in cluster_ids:
            cluster_means = df[df['Cluster_ID'] == cluster_id][top_numeric_features].mean()
            numeric_summary_df[f'Cluster {cluster_id}'] = cluster_means
        summary_parts.append(numeric_summary_df)

    top_categorical_features = [f for f in top_feature_names if f in cat_cols]
    for feature in top_categorical_features:
        cat_summary_df = pd.DataFrame({ 'Global': df[feature].value_counts(normalize=True) * 100 })
        for cluster_id in cluster_ids:
            cat_summary_df[f'Cluster {cluster_id}'] = df[df['Cluster_ID'] == cluster_id][feature].value_counts(normalize=True) * 100
        cat_summary_df = cat_summary_df.fillna(0)
        cat_summary_df.index = [f"{feature}_%_{cat}" for cat in cat_summary_df.index]
        summary_parts.append(cat_summary_df)

    if summary_parts:
        final_summary_df = pd.concat(summary_parts).fillna(0)
        final_summary_df.round(2).to_csv('output/cluster_summary_comparison.csv')
        print("Exported full summary data to 'cluster_summary_comparison.csv'")

    # --- Generate one profile chart per cluster ---
    print("\n--- Generating profile chart for each cluster ---")
    sns.set_theme(style="whitegrid")

    for cluster_id in cluster_ids:
        cluster_data = df[df['Cluster_ID'] == cluster_id]
        other_data = df[df['Cluster_ID'] != cluster_id] # Define other_data once per cluster
        plot_data = []

        for feature in top_numeric_features:
            plot_data.append({'Feature': feature, 'Value': cluster_data[feature].mean(), 'Type': f'Cluster {cluster_id}'})
            plot_data.append({'Feature': feature, 'Value': global_means[feature], 'Type': 'Global'})

        for feature in top_categorical_features:
            cluster_dist = cluster_data[feature].value_counts(normalize=True)
            if cluster_dist.empty: continue
            dominant_category = cluster_dist.idxmax()
            feature_label = f"{feature}_%_{dominant_category}"

            plot_data.append({'Feature': feature_label, 'Value': cluster_dist.max() * 100, 'Type': f'Cluster {cluster_id}'})
            plot_data.append({'Feature': feature_label, 'Value': (df[feature].value_counts(normalize=True).get(dominant_category, 0)) * 100, 'Type': 'Global'})

        if not plot_data: continue
        plot_df = pd.DataFrame(plot_data)

        # Get the order of features for plotting and annotation
        feature_order = plot_df['Feature'].unique()

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Value', y='Feature', hue='Type', data=plot_df, palette=['#4c72b0', '#c44e52'], order=feature_order)
        ax.bar_label(ax.containers[0], fmt='%.2f', padding=3)
        ax.bar_label(ax.containers[1], fmt='%.2f', padding=3)
        ax.set_title(f'Profile of Cluster {cluster_id} (Size: {len(cluster_data)}) vs. Global Average', fontsize=16, pad=20)
        ax.set_xlabel('Value / Percentage')
        ax.set_ylabel('Feature')

        ax.set_xlim(right=ax.get_xlim()[1] * 1.15) # Adjust x-limit to make space for bar labels
        plt.tight_layout()
        filename = f'plots/cluster_{cluster_id}_profile.png'
        plt.savefig(filename)
        plt.close()
        print(f"Exported profile chart to '{filename}'")