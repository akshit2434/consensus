import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from pathlib import Path
import seaborn as sns
from chat_analyzer import load_mock_data, generate_embeddings, perform_clustering

def load_cached_data():
    """Load cached embeddings and messages"""
    try:
        with open("embeddings_cache.pkl", 'rb') as f:
            cached_data = pickle.load(f)
        
        messages = load_mock_data('mock_stream_chat.json')
        return cached_data['embeddings'], messages
    except FileNotFoundError:
        print("No cached embeddings found. Run chat_analyzer.py first.")
        return None, None

def reduce_dimensions(embeddings, method='pca'):
    """Reduce embeddings to 2D for visualization"""
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        explained_variance = reducer.explained_variance_ratio_
        print(f"PCA explained variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
        return reduced
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        return reducer.fit_transform(embeddings)

def plot_clusters(embeddings, messages, method='pca', save_plot=True):
    """Visualize clusters in 2D space"""
    # Perform clustering
    cluster_labels, clusterer = perform_clustering(embeddings)
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(embeddings, method)
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Create color map
    unique_labels = set(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Outliers in black
            mask = cluster_labels == label
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Outliers')
        else:
            mask = cluster_labels == label
            cluster_size = sum(mask)
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                       c=[color], s=60, alpha=0.8, label=f'Cluster {label} ({cluster_size} msgs)')
    
    plt.title(f'Chat Message Clusters - {method.upper()} Visualization\n'
              f'Total: {len(messages)} messages, Clusters: {len(unique_labels)-1}, '
              f'Outliers: {sum(1 for l in cluster_labels if l == -1)}')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'cluster_visualization_{method}.png', dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualization to cluster_visualization_{method}.png")
    
    plt.show()
    
    return cluster_labels, reduced_embeddings

def analyze_cluster_details(embeddings, messages, cluster_labels):
    """Print detailed cluster analysis"""
    unique_labels = set(cluster_labels)
    valid_clusters = [l for l in unique_labels if l != -1]
    outliers_count = sum(1 for l in cluster_labels if l == -1)
    
    print("\n" + "="*60)
    print("🔍 DETAILED CLUSTER ANALYSIS")
    print("="*60)
    
    for cluster_id in sorted(valid_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_messages = [messages[i] for i in range(len(messages)) if cluster_mask[i]]
        cluster_size = len(cluster_messages)
        
        print(f"\n📌 Cluster {cluster_id} ({cluster_size} messages):")
        print("-" * 40)
        
        # Show all messages in smaller clusters, sample from larger ones
        if cluster_size <= 8:
            for i, msg in enumerate(cluster_messages, 1):
                print(f"  {i}. \"{msg}\"")
        else:
            print("  First 5 messages:")
            for i, msg in enumerate(cluster_messages[:5], 1):
                print(f"  {i}. \"{msg}\"")
            print(f"  ... and {cluster_size - 5} more messages")
    
    if outliers_count > 0:
        print(f"\n❌ Outliers ({outliers_count} messages):")
        print("-" * 40)
        outlier_messages = [messages[i] for i in range(len(messages)) if cluster_labels[i] == -1]
        
        if outliers_count <= 10:
            for i, msg in enumerate(outlier_messages, 1):
                print(f"  {i}. \"{msg}\"")
        else:
            print("  First 10 outlier messages:")
            for i, msg in enumerate(outlier_messages[:10], 1):
                print(f"  {i}. \"{msg}\"")
            print(f"  ... and {outliers_count - 10} more outliers")

def compare_clustering_parameters():
    """Compare different clustering parameters"""
    embeddings, messages = load_cached_data()
    if embeddings is None:
        return
    
    print("\n🔬 COMPARING CLUSTERING PARAMETERS")
    print("="*50)
    
    parameters = [
        {'min_cluster_size': 2, 'min_samples': 1, 'cluster_selection_epsilon': 0.1},
        {'min_cluster_size': 3, 'min_samples': 1, 'cluster_selection_epsilon': 0.1},
        {'min_cluster_size': 2, 'min_samples': 1, 'cluster_selection_epsilon': 0.2},
        {'min_cluster_size': 4, 'min_samples': 2, 'cluster_selection_epsilon': 0.1},
    ]
    
    for i, params in enumerate(parameters, 1):
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(**params, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)
        
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_outliers = sum(1 for l in labels if l == -1)
        outlier_ratio = n_outliers / len(labels)
        
        print(f"\nTest {i}: {params}")
        print(f"  Clusters: {n_clusters}, Outliers: {n_outliers} ({outlier_ratio:.1%})")

def main():
    print("🎯 Chat Cluster Visualizer")
    print("="*40)
    
    # Load data
    embeddings, messages = load_cached_data()
    if embeddings is None:
        print("Please run chat_analyzer.py first to generate embeddings.")
        return
    
    print(f"📊 Loaded {len(messages)} messages with {embeddings.shape[1]}D embeddings")
    
    # Compare clustering parameters
    compare_clustering_parameters()
    
    # Visualize with PCA
    print(f"\n📈 Creating PCA visualization...")
    cluster_labels_pca, _ = plot_clusters(embeddings, messages, method='pca')
    
    # Visualize with t-SNE
    print(f"\n📈 Creating t-SNE visualization...")
    cluster_labels_tsne, _ = plot_clusters(embeddings, messages, method='tsne')
    
    # Detailed analysis
    analyze_cluster_details(embeddings, messages, cluster_labels_pca)
    
    print(f"\n✅ Visualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    # Add matplotlib to requirements
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Installing visualization dependencies...")
        import subprocess
        subprocess.run(["pip", "install", "matplotlib", "seaborn"], check=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    
    main()