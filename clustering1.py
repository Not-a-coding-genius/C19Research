import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform
import fastcluster
import warnings
from tqdm import tqdm
import argparse
import os
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

class ResearchPaperClustering:
    def __init__(self, file_path=None, sample_size=5000, random_state=42):
        """Initialize the clustering pipeline with configurable parameters."""
        self.file_path = file_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        self.df_sample = None
        self.X = None
        self.vectorizer = None
        self.pca = None
        self.clustering = None
        
        # Custom stopwords
        self.custom_stopwords = set([
            "the", "and", "is", "in", "to", "of", "for", "on", "with", "as", 
            "this", "that", "a", "an", "it", "at", "by", "from", "which", 
            "be", "or", "are", "been", "has", "have", "had", "can", "will", 
            "would", "should", "could", "may", "might", "must", "shall"
        ])
        
    def load_data(self):
        """Load data from CSV file."""
        print("📂 Loading data from:", self.file_path)
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Display basic dataset information
            print("\n📊 Dataset Overview:")
            print(f"Missing values in abstract: {self.df['abstract_summary'].isna().sum()}")
            print(f"Countries represented: {self.df['country'].nunique()}")
            print(f"Journals represented: {self.df['journal'].nunique()}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
            
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if pd.isna(text): 
            return ""
        # Remove special characters and convert to lowercase
        text = re.sub(r'\W+', ' ', str(text)).lower()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove stopwords
        words = [word for word in text.split() if word not in self.custom_stopwords and len(word) > 2]
        return " ".join(words)
    
    def preprocess_data(self):
        """Preprocess data including text cleaning and sampling."""
        print("\n🔄 Preprocessing data...")
        
        # Apply text cleaning
        tqdm.pandas(desc="Cleaning abstracts")
        self.df['clean_abstract'] = self.df['abstract_summary'].progress_apply(self.clean_text)
        
        if 'body_text' in self.df.columns:
            tqdm.pandas(desc="Cleaning body text")
            self.df['clean_body'] = self.df['body_text'].progress_apply(self.clean_text)
            self.df['combined_text'] = self.df['clean_abstract'] + " " + self.df['clean_body']
            # Calculate word count for metadata
            self.df['word_count'] = self.df['body_text'].apply(lambda x: len(str(x).split()))
        else:
            self.df['combined_text'] = self.df['clean_abstract']
            self.df['word_count'] = self.df['abstract_summary'].apply(lambda x: len(str(x).split()))
        
        # Sample data to manageable size
        if self.sample_size > 0 and self.sample_size < len(self.df):
            self.df_sample = self.df.sample(n=self.sample_size, random_state=self.random_state)
            print(f"📌 Sampled {self.sample_size} rows from dataset")
        else:
            self.df_sample = self.df
            print(f"📌 Using all {len(self.df)} rows from dataset")
            
        # Handle missing values in important columns
        for col in ['country', 'journal']:
            missing = self.df_sample[col].isna().sum()
            if missing > 0:
                print(f"⚠️ Found {missing} missing values in '{col}'. Filling with 'unknown'")
                self.df_sample[col] = self.df_sample[col].fillna('unknown')
                
        # Handle missing values in word_count
        if self.df_sample['word_count'].isna().sum() > 0:
            print(f"⚠️ Found {self.df_sample['word_count'].isna().sum()} missing values in 'word_count'. Filling with median value.")
            self.df_sample['word_count'] = self.df_sample['word_count'].fillna(self.df_sample['word_count'].median())
    
    def create_text_features(self):
        """Create text features using TF-IDF vectorization."""
        print("\n🔤 Creating text features using TF-IDF...")
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            min_df=5,  # Ignore terms that appear in less than 5 documents
            max_df=0.7  # Ignore terms that appear in more than 70% of documents
        )
        
        X_text = self.vectorizer.fit_transform(self.df_sample['combined_text'])
        print(f"✅ Created {X_text.shape[1]} text features")
        
        return X_text.toarray()
    
    def create_metadata_features(self):
        """Create metadata features with proper encoding."""
        print("\n📋 Creating metadata features...")
        
        # Handle categorical features
        metadata_df = pd.DataFrame()
        
        # Encode categorical variables
        for col in ['country', 'journal']:
            self.df_sample[col] = self.df_sample[col].astype(str)
            encoder = LabelEncoder()
            metadata_df[col] = encoder.fit_transform(self.df_sample[col])
            print(f"✅ Encoded '{col}' with {encoder.classes_.shape[0]} unique values")
        
        # Add numerical features
        metadata_df['word_count'] = self.df_sample['word_count']
        
        # Check for and handle any remaining NaN values
        for col in metadata_df.columns:
            if metadata_df[col].isna().any():
                print(f"⚠️ Found NaN values in '{col}' after preprocessing. Imputing values.")
                # For numerical columns, impute with median
                if np.issubdtype(metadata_df[col].dtype, np.number):
                    imputer = SimpleImputer(strategy='median')
                # For categorical columns, impute with most frequent value
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                
                metadata_df[col] = imputer.fit_transform(metadata_df[[col]])
        
        # Scale numerical features
        scaler = StandardScaler()
        X_meta = scaler.fit_transform(metadata_df)
        print(f"✅ Created {X_meta.shape[1]} metadata features")
        
        # Final check for NaNs
        if np.isnan(X_meta).any():
            print("⚠️ NaNs still present after preprocessing. Applying final imputation.")
            imputer = SimpleImputer(strategy='median')
            X_meta = imputer.fit_transform(X_meta)
        
        return X_meta
    
    def perform_clustering(self, method='text', n_clusters=5):
        """Perform clustering using the specified method."""
        print(f"\n🔍 Performing clustering using {method.upper()} features...")
        
        # Generate features based on method
        if method == 'text':
            self.X = self.create_text_features()
        elif method == 'metadata':
            self.X = self.create_metadata_features()
        elif method == 'hybrid':
            X_text = self.create_text_features()
            X_meta = self.create_metadata_features()
            # Combine text and metadata features
            self.X = np.hstack((X_text, X_meta))
            print(f"✅ Combined features shape: {self.X.shape}")
        else:
            print("❌ Invalid method. Using text features by default.")
            self.X = self.create_text_features()
        
        # Final check for NaNs
        if np.isnan(self.X).any():
            nan_count = np.isnan(self.X).sum()
            print(f"⚠️ Found {nan_count} NaN values in feature matrix. Applying imputation.")
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Apply dimensionality reduction for high-dimensional feature sets
        if self.X.shape[1] > 50:
            n_components = min(50, self.X.shape[1], self.X.shape[0])
            print(f"🔄 Reducing dimensions from {self.X.shape[1]} to {n_components} using PCA...")
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.X = self.pca.fit_transform(self.X)
            explained_var = sum(self.pca.explained_variance_ratio_) * 100
            print(f"✅ PCA explains {explained_var:.2f}% of variance")
        
        # Apply hierarchical clustering
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='euclidean', 
            linkage='ward'
        )
        
        self.df_sample['cluster'] = self.clustering.fit_predict(self.X)
        
        # Analyze clusters
        cluster_counts = self.df_sample['cluster'].value_counts().sort_index()
        print("\n📊 Cluster distribution:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} papers ({count/len(self.df_sample)*100:.2f}%)")
            
        # Save the clustering results
        self.save_results(method)
        
        return self.df_sample['cluster']
    
    def save_results(self, method):
        """Save clustering results to CSV file."""
        output_dir = "clustering_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cluster assignments
        output_file = f"{output_dir}/cluster_results_{method}_{self.sample_size}.csv"
        self.df_sample[['title', 'journal', 'country', 'cluster']].to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")
        
        # Save model components if available
        if self.vectorizer:
            joblib.dump(self.vectorizer, f"{output_dir}/vectorizer_{method}.pkl")
        if self.pca:
            joblib.dump(self.pca, f"{output_dir}/pca_{method}.pkl")
        if self.clustering:
            joblib.dump(self.clustering, f"{output_dir}/clustering_{method}.pkl")
    
    def visualize_dendrogram(self, max_samples=500, output_path="clustering_results/dendrogram.png"):
    # Use a subset of data for visualization if necessary
        X_subset = self.X[:min(max_samples, len(self.X))]
        
        plt.figure(figsize=(12, 8))
        
        # Calculate linkage matrix using fastcluster for better performance
        print(f"📊 Calculating linkage matrix for {len(X_subset)} samples...")
        linkage_matrix = fastcluster.linkage_vector(X_subset, method='ward')
        
        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            truncate_mode="level",
            p=5,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
            show_leaf_counts=True,
            no_labels=True
        )
        plt.title("🔗 Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to prevent display in notebook environments

    def visualize_pca_scatter(self, output_path="clustering_results/pca_scatter.png"):
        """Visualize clusters in 2D PCA space."""
        # Reduce to 2D for visualization
        if self.X.shape[1] > 2:
            pca_viz = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca_viz.fit_transform(self.X)
        else:
            X_2d = self.X
        
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            x=X_2d[:, 0], 
            y=X_2d[:, 1], 
            hue=self.df_sample['cluster'],
            palette="tab10", 
            s=70,
            alpha=0.7
        )
        
        # Add labels for centroids
        for cluster_id in sorted(self.df_sample['cluster'].unique()):
            mask = self.df_sample['cluster'] == cluster_id
            centroid_x = X_2d[mask, 0].mean()
            centroid_y = X_2d[mask, 1].mean()
            plt.text(
                centroid_x, centroid_y, 
                f'Cluster {cluster_id}',
                fontsize=12, 
                fontweight='bold',
                ha='center', 
                va='center',
                bbox=dict(facecolor='white', alpha=0.6)
            )
            
        plt.title("🔍 Hierarchical Clustering Results (PCA Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title="Cluster", loc='best')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to prevent display in notebook environments

    def visualize_heatmap(self, max_samples=200, output_path="clustering_results/similarity_heatmap.png"):
        """Visualize similarity heatmap between samples."""
        # Use a subset of data for visualization
        X_subset = self.X[:min(max_samples, len(self.X))]
        
        # Create a subset dataframe that exactly matches the subset of X we're visualizing
        subset_df = self.df_sample.iloc[:len(X_subset)].copy()
        
        # Calculate distance matrix
        print(f"📊 Calculating distance matrix for {len(X_subset)} samples...")
        dist_matrix = squareform(pdist(X_subset, metric="euclidean"))
        
        # Sort by cluster for better visualization
        # Get indices within the subset (not the original dataframe indices)
        cluster_order = subset_df.sort_values('cluster').index
        
        # Convert to positional indices (0 to len(subset)-1)
        positional_indices = [subset_df.index.get_loc(idx) for idx in cluster_order]
        
        # Use positional indices to reorder the distance matrix
        dist_matrix_sorted = dist_matrix[np.ix_(positional_indices, positional_indices)]
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            dist_matrix_sorted, 
            cmap="viridis_r",
            xticklabels=False,
            yticklabels=False
        )
        plt.title("🔥 Sample Similarity Heatmap (sorted by cluster)")
        plt.xlabel("Data Points")
        plt.ylabel("Data Points")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to prevent display in notebook environments
    

    
    def analyze_clusters(self):
        """Analyze cluster contents to determine common themes."""
        print("\n📝 Analyzing clusters for common themes...")
        
        # Group by cluster
        for cluster_id in sorted(self.df_sample['cluster'].unique()):
            cluster_df = self.df_sample[self.df_sample['cluster'] == cluster_id]
            
            print(f"\n📌 Cluster {cluster_id} ({len(cluster_df)} papers):")
            
            # Get most common journals
            top_journals = cluster_df['journal'].value_counts().head(3)
            print(f"  📚 Top journals: {', '.join(top_journals.index)}")
            
            # Get most common countries
            top_countries = cluster_df['country'].value_counts().head(3)
            print(f"  🌍 Top countries: {', '.join(top_countries.index)}")
            
            # Get most representative titles (sample)
            print(f"  📑 Sample titles:")
            for title in cluster_df['title'].sample(min(3, len(cluster_df))).values:
                print(f"    - {title}")
            
            # Extract key terms if vectorizer is available
            if self.vectorizer and hasattr(self.vectorizer, 'get_feature_names_out'):
                try:
                    # Get feature names
                    feature_names = self.vectorizer.get_feature_names_out()
                    
                    # If we used PCA, we need a different approach to get important terms
                    if self.pca is not None:
                        # We'll use the raw TF-IDF matrix
                        tfidf_matrix = self.vectorizer.transform(cluster_df['combined_text'])
                        
                        # Get average TF-IDF scores for this cluster
                        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                        
                        # Get top terms
                        top_indices = avg_tfidf.argsort()[-10:][::-1]
                        top_terms = [feature_names[i] for i in top_indices]
                    else:
                        # Get the centroid of this cluster
                        cluster_center = np.mean(self.X[self.df_sample['cluster'] == cluster_id], axis=0)
                        
                        # Get top terms (only applicable if not using PCA or directly using TF-IDF)
                        if cluster_center.shape[0] == len(feature_names):
                            top_indices = cluster_center.argsort()[-10:][::-1]
                            top_terms = [feature_names[i] for i in top_indices]
                        else:
                            # Can't directly map to features, use TF-IDF approach
                            tfidf_matrix = self.vectorizer.transform(cluster_df['combined_text'])
                            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                            top_indices = avg_tfidf.argsort()[-10:][::-1]
                            top_terms = [feature_names[i] for i in top_indices]
                    
                    print(f"  🔤 Key terms: {', '.join(top_terms)}")
                except Exception as e:
                    print(f"  ⚠️ Could not extract key terms: {str(e)}")
    
    def run_interactive(self):
        """Run the clustering pipeline interactively."""
        if not self.load_data():
            return
            
        self.preprocess_data()
        
        while True:
            print("\n📊 Choose Clustering Method:")
            print("1️⃣ - Text-Based Clustering (Abstract + Body Text)")
            print("2️⃣ - Metadata-Based Clustering (Country, Journal, Word Count)")
            print("3️⃣ - Hybrid Clustering (Text + Metadata)")
            print("4️⃣ - Exit")

            try:
                choice = int(input("➡ Enter 1, 2, 3, or 4: "))
            except ValueError:
                print("❌ Please enter a valid number.")
                continue

            if choice == 4:
                print("✅ Exiting the program.")
                break
                
            # Get number of clusters
            try:
                n_clusters = int(input("➡ Enter number of clusters (2-20): "))
                n_clusters = max(2, min(20, n_clusters))  # Ensure between 2 and 20
            except ValueError:
                n_clusters = 5
                print("⚠️ Using default value of 5 clusters")

            # Perform clustering based on choice
            if choice == 1:
                self.perform_clustering(method='text', n_clusters=n_clusters)
            elif choice == 2:
                self.perform_clustering(method='metadata', n_clusters=n_clusters)
            elif choice == 3:
                self.perform_clustering(method='hybrid', n_clusters=n_clusters)
            else:
                print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
                continue
                
            # Analyze clusters
            self.analyze_clusters()

            # Visualization menu
            while True:
                print("\n📊 Choose a Visualization Method:")
                print("1️⃣ - Dendrogram")
                print("2️⃣ - PCA Scatter Plot")
                print("3️⃣ - Cluster Heatmap")
                print("4️⃣ - Go Back to Clustering Selection")

                try:
                    viz_choice = int(input("➡ Enter 1, 2, 3, or 4: "))
                except ValueError:
                    print("❌ Please enter a valid number.")
                    continue

                if viz_choice == 1:
                    max_samples = int(input("➡ Enter maximum samples for dendrogram (50-1000): ") or "500")
                    max_samples = max(50, min(1000, max_samples))
                    self.visualize_dendrogram(max_samples=max_samples)
                elif viz_choice == 2:
                    self.visualize_pca_scatter()
                elif viz_choice == 3:
                    max_samples = int(input("➡ Enter maximum samples for heatmap (50-500): ") or "200")
                    max_samples = max(50, min(500, max_samples))
                    self.visualize_heatmap(max_samples=max_samples)
                elif viz_choice == 4:
                    print("🔄 Going back to clustering selection...")
                    break
                else:
                    print("❌ Invalid choice. Please select 1, 2, 3, or 4.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Research Paper Clustering Tool')
    parser.add_argument('--file', '-f', type=str, required=False,
                        default=r"E:\DM PROJ\DM proj\cord_data\df_200k.csv\df_200k.csv",
                        help='Path to the CSV file containing the papers data')
    parser.add_argument('--sample', '-s', type=int, default=5000,
                        help='Number of papers to sample for analysis (default: 5000)')
    parser.add_argument('--method', '-m', type=str, choices=['text', 'metadata', 'hybrid'],
                        default=None, help='Clustering method to use (default: interactive mode)')
    parser.add_argument('--clusters', '-c', type=int, default=5,
                        help='Number of clusters to create (default: 5)')
    parser.add_argument('--random_state', '-r', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize clustering pipeline
    clustering = ResearchPaperClustering(
        file_path=args.file,
        sample_size=args.sample,
        random_state=args.random_state
    )
    
    # Load and preprocess data
    if not clustering.load_data():
        print("❌ Exiting due to data loading error.")
        exit(1)
        
    clustering.preprocess_data()
    
    # Run in command line mode if method is specified
    if args.method:
        clustering.perform_clustering(method=args.method, n_clusters=args.clusters)
        clustering.analyze_clusters()
        clustering.visualize_pca_scatter()
        print("✅ Clustering complete. Results saved in 'clustering_results' directory.")
    else:
        # Run in interactive mode
        clustering.run_interactive()