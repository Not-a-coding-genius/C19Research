import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import requests
import joblib
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
import importlib.util
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Import the ResearchPaperClustering class from clustering1.py
spec = importlib.util.spec_from_file_location("clustering1", "clustering1.py")
clustering1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clustering1)
ResearchPaperClustering = clustering1.ResearchPaperClustering

# Define the dataset path
DEFAULT_DATASET_PATH = r"E:\DMPROJ1\df_200k.csv"

# Define function to load an image file as base64 for display
def get_image_as_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to call NVIDIA API for cluster insights
def get_ai_insights(cluster_data, cluster_id, model="nvidia/llama-3.1-nemotron-ultra-253b-v1", temperature=0.7):
    """
    Generate AI-driven research insights for a given cluster of academic papers.

    Args:
        cluster_data (pd.DataFrame): DataFrame containing 'cluster', 'title', 'journal', and 'country' columns.
        cluster_id (int or str): Cluster ID to analyze.
        model (str): Model name for the NVIDIA API (default: LLaMA 3.1 Ultra).
        temperature (float): Sampling temperature for generation (default: 0.7).

    Returns:
        str: Formatted string containing AI-generated insights or error message.
    """

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return "⚠️ NVIDIA API key not found. Please add it to your .env file as NVIDIA_API_KEY=your_key_here"

    try:
        # Filter and sample data for the specific cluster
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        sample_size = min(10, len(cluster_subset))
        if sample_size == 0:
            return f"⚠️ No data found for cluster ID: {cluster_id}"
        cluster_sample = cluster_subset.sample(sample_size)

        # Extract top journals and countries
        top_journals = ', '.join(cluster_sample['journal'].value_counts().head(3).index.tolist())
        top_countries = ', '.join(cluster_sample['country'].value_counts().head(3).index.tolist())

        # Format titles for prompt
        sample_titles = "\n".join([f"- {title}" for title in cluster_sample['title'].tolist()])

        prompt = f"""
You are an expert research analyst diving into a cluster of academic papers.

Your mission is to:
1. **Distill** the overarching research focus by examining the titles.
2. **Spot patterns** in the journals and countries — are there shared interests, biases, or regional strengths?
3. **Name the cluster** with a short, punchy, and descriptive label — think like you're naming a research theme in a conference program.

Be sharp and concise in your response. Here's the structure to follow:

**Theme:** <What are these papers mainly about?>
**Patterns:** <What do the journals and countries tell us? Any trends, biases, or specializations?>
**Cluster Name:** <A short, creative name summarizing the cluster>

---

**Paper Titles:**
{sample_titles}

**Top Journals:** {top_journals}
**Top Countries:** {top_countries}
"""

        # NVIDIA API settings
        API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful research analyst that provides concise insights about academic paper clusters."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": 512
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return f"❌ Error calling NVIDIA API: Status {response.status_code}\nMessage: {response.text}"

        try:
            result = response.json()
        except json.JSONDecodeError:
            return f"❌ Error parsing JSON response. Raw output:\n{response.text[:200]}..."

        # Extract content
        if 'choices' in result and result['choices']:
            content = result['choices'][0].get('message', {}).get('content', '').strip()
            if content:
                return f"## Cluster {cluster_id} Analysis\n\n{content}"
            else:
                return "⚠️ Received empty response content from the API."
        else:
            return f"⚠️ Unexpected API response format:\n{json.dumps(result, indent=2)[:300]}..."

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# Function to extract method and sample size from filename
def parse_cluster_filename(filename):
    pattern = r"cluster_results_(\w+)_(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        method, sample_size = match.groups()
        return method, int(sample_size)
    return None, None

# Function to get a nice display name for a clustering result file
def get_display_name(filename):
    method, sample_size = parse_cluster_filename(filename)
    if method is None:
        return filename
    
    method_names = {
        "text": "Text-Based",
        "metadata": "Metadata-Based",
        "hybrid": "Hybrid"
    }
    
    method_display = method_names.get(method, method.capitalize())
    return f"{method_display} Clustering ({sample_size} samples)"

# Define the main app
def main():
    st.set_page_config(
        page_title="Research Paper Clustering Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Create necessary directories
    os.makedirs("clustering_results", exist_ok=True)

    # Custom CSS for dark theme with BALANCED text colors
    st.markdown("""
    <style>
    /* Main dark theme */
    .stApp {
        background-color: #121212;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        text-shadow: 0px 0px 10px rgba(0,191,255,0.5);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #FFFFFF;
    }
    
    /* Cards and containers */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1E1E1E;
        margin-bottom: 1rem;
        border: 1px solid #3D3D3D;
    }
    
    .cluster-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1E1E1E;
        margin-bottom: 1rem;
        border-left: 5px solid #00BFFF;
    }
    
    /* Fix markdown text only (not inputs/selects) */
    .st-emotion-cache-eczf16 p, 
    .st-emotion-cache-eczf16 li,
    .st-emotion-cache-eczf16 h1, 
    .st-emotion-cache-eczf16 h2, 
    .st-emotion-cache-eczf16 h3, 
    .st-emotion-cache-eczf16 h4 {
        color: #FFFFFF !important;
    }
    
    /* Static content headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Text color for static paragraph content */
    .stMarkdown p, .stMarkdown li {
        color: #FFFFFF !important;
    }
    
    /* Override only specific static text elements */
    .stHeader, .stTitle, .stSubheader, .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* DO NOT override interactive element text colors */
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #FFFFFF;
    }
    
    /* Fix for code blocks */
    code {
        color: #00BFFF !important;
        background-color: #2D2D2D !important;
    }
    
    /* Fix Plotly chart text */
    .js-plotly-plot .plotly .main-svg text {
        fill: #FFFFFF !important;
    }
    
    /* Sidebar header text color */
    .sidebar .sidebar-content .block-container h1,
    .sidebar .sidebar-content .block-container h2,
    .sidebar .sidebar-content .block-container h3 {
        color: #FFFFFF !important;
    }
    
    /* Keep select box text black for readability */
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* Fix success message text */
    .stSuccess {
        color: #000000 !important;
    }
    
    /* Fix info message text */
    .stInfo {
        color: #000000 !important;
    }
    
    /* Fix warning message text */
    .stWarning {
        color: #000000 !important;
    }
    
    /* Fix error message text */
    .stError {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # App title
    st.markdown('<p class="main-header">📚 Research Paper Clustering Dashboard</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.header("Configuration")
        
        # API key info
        api_key = os.getenv("NVIDIA_API_KEY")
        if api_key:
            st.success("✅ API key detected")
        else:
            st.warning("⚠️API key not found in .env file")
            st.info("Create a .env file with NVIDIA_API_KEY=your_key_here")
        
        
        # Sections for different operations
        st.subheader("Operations")
        app_mode = st.radio(
            "Select Mode",
            ["View Existing Results", "Run New Analysis"]
        )
        
        if app_mode == "Run New Analysis":
            # Dataset path
            dataset_path = st.text_input("Dataset Path", value=DEFAULT_DATASET_PATH)
            
            # Clustering parameters
            sample_size = st.slider("Sample Size", min_value=100, max_value=20000, value=5000, step=100)
            clustering_method = st.selectbox(
                "Clustering Method",
                ["text", "metadata", "hybrid"],
                format_func=lambda x: {
                    "text": "Text-Based Clustering",
                    "metadata": "Metadata-Based Clustering",
                    "hybrid": "Hybrid Clustering (Text + Metadata)"
                }.get(x)
            )
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=5)
            
            run_analysis = st.button("Run Clustering Analysis")
            
            if run_analysis:
                if not os.path.exists(dataset_path):
                    st.error(f"Dataset not found at: {dataset_path}")
                else:
                    # Run analysis
                    with st.spinner("Running clustering analysis..."):
                        # Create clustering instance
                        clustering = ResearchPaperClustering(
                            file_path=dataset_path,
                            sample_size=sample_size,
                            random_state=42
                        )
                        
                        # Run analysis
                        if clustering.load_data():
                            clustering.preprocess_data()
                            clustering.perform_clustering(method=clustering_method, n_clusters=n_clusters)
                            clustering.analyze_clusters()
                            
                            # Generate visualizations with unique filenames based on method and sample size
                            with st.expander("Generating Visualizations..."):
                                # Save with unique names
                                pca_filename = f"pca_scatter_{clustering_method}_{sample_size}.png"
                                dendrogram_filename = f"dendrogram_{clustering_method}_{sample_size}.png"
                                heatmap_filename = f"similarity_heatmap_{clustering_method}_{sample_size}.png"
                                
                                clustering.visualize_pca_scatter(output_path=f"clustering_results/{pca_filename}")
                                st.success(f"✅ Generated PCA scatter plot: {pca_filename}")
                                
                                clustering.visualize_dendrogram(max_samples=500, 
                                                               output_path=f"clustering_results/{dendrogram_filename}")
                                st.success(f"✅ Generated dendrogram: {dendrogram_filename}")
                                
                                clustering.visualize_heatmap(max_samples=200,
                                                           output_path=f"clustering_results/{heatmap_filename}")
                                st.success(f"✅ Generated similarity heatmap: {heatmap_filename}")
                            
                            st.success("Analysis complete! Switch to 'View Existing Results' to explore.")
                        else:
                            st.error("Error loading data. Please check your dataset file.")

    # Main area - different based on the selected mode
    if app_mode == "View Existing Results":
        # Available clustering results
        clustering_results_path = "clustering_results"
        clustering_results_files = [f for f in os.listdir(clustering_results_path) 
                                   if f.startswith("cluster_results_") and f.endswith(".csv")]
        
        if not clustering_results_files:
            st.warning("No clustering results found. Please run a new analysis first.")
            return
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Cluster Explorer", 
            "PCA Visualization", 
            "Dendrogram", 
            "Similarity Heatmap"
        ])
        
        # Tab 1: Cluster Explorer
        with tab1:
            st.markdown('<p class="sub-header">🔍 Explore Clusters</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Select which clustering results to view - FIXED VERSION
                selected_file = st.selectbox(
                    "Select Clustering Results",
                    clustering_results_files,
                    format_func=get_display_name  # Use the improved formatting function
                )
                
                # Extract method and sample size for visualization filenames
                method, sample_size = parse_cluster_filename(selected_file)
                
                # Load selected results
                try:
                    cluster_data = pd.read_csv(f"{clustering_results_path}/{selected_file}")
                    
                    # Get cluster distribution
                    cluster_counts = cluster_data['cluster'].value_counts().sort_index()
                    
                    # Show cluster selection
                    st.subheader("Select Cluster")
                    selected_cluster = st.selectbox(
                        "Cluster ID",
                        sorted(cluster_data['cluster'].unique()),
                        format_func=lambda x: f"Cluster {x} ({cluster_counts[x]} papers)"
                    )
                    
                    # Show distribution chart - IMPROVED COLORS FOR READABILITY
                    st.subheader("Cluster Distribution")
                    fig = px.bar(
                        x=cluster_counts.index, 
                        y=cluster_counts.values,
                        labels={'x': 'Cluster', 'y': 'Number of Papers'},
                        color=cluster_counts.index,
                        color_continuous_scale=px.colors.sequential.Turbo  # Vibrant scale
                    )
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        template="plotly_dark",
                        paper_bgcolor="#1E1E1E",
                        plot_bgcolor="#1E1E1E",
                        font=dict(color="#FFFFFF", size=14)  # Larger, white text
                    )
                    # Make axis labels and tick values more visible
                    fig.update_xaxes(title_font=dict(size=16, color="#FFFFFF"), 
                                    tickfont=dict(size=14, color="#FFFFFF"))
                    fig.update_yaxes(title_font=dict(size=16, color="#FFFFFF"), 
                                    tickfont=dict(size=14, color="#FFFFFF"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to get AI insights
                    if st.button("Generate AI Insights for Selected Cluster"):
                        with st.spinner("Getting AI insights..."):
                            insights = get_ai_insights(cluster_data, selected_cluster)
                            st.session_state['current_insights'] = insights
                
                except Exception as e:
                    st.error(f"Error loading cluster data: {str(e)}")
                    return
            
            with col2:
                if 'cluster_data' in locals():
                    # Filter data for the selected cluster
                    cluster_df = cluster_data[cluster_data['cluster'] == selected_cluster]
                    
                    # Display cluster information in a card
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h3>Cluster {selected_cluster} Details</h3>
                        <p><b>Number of Papers:</b> {len(cluster_df)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top journals and countries
                    col_j, col_c = st.columns(2)
                    
                    with col_j:
                        st.subheader("Top Journals")
                        journal_counts = cluster_df['journal'].value_counts().head(5)
                        # IMPROVED CHART FOR READABILITY
                        fig = px.pie(
                            values=journal_counts.values,
                            names=journal_counts.index,
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Bold  # Better distinct colors
                        )
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            template="plotly_dark",
                            paper_bgcolor="#1E1E1E",
                            plot_bgcolor="#1E1E1E",
                            font=dict(color="#FFFFFF", size=14)  # Larger font
                        )
                        # Make legend text more visible
                        fig.update_traces(textinfo='percent+label', textfont_size=14)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_c:
                        st.subheader("Top Countries")
                        country_counts = cluster_df['country'].value_counts().head(5)
                        # IMPROVED CHART FOR READABILITY
                        fig = px.pie(
                            values=country_counts.values,
                            names=country_counts.index,
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Plotly  # Better distinct colors
                        )
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            template="plotly_dark",
                            paper_bgcolor="#1E1E1E",
                            plot_bgcolor="#1E1E1E",
                            font=dict(color="#FFFFFF", size=14)  # Larger font
                        )
                        # Make legend text more visible
                        fig.update_traces(textinfo='percent+label', textfont_size=14)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample titles
                    st.subheader("Sample Paper Titles")
                    for title in cluster_df['title'].sample(min(10, len(cluster_df))).values:
                        st.markdown(f"- {title}")
                    
                    # Display AI insights if available
                    if 'current_insights' in st.session_state:
                        st.subheader("🤖 AI Insights")
                        st.markdown(st.session_state['current_insights'])
                
# Tab 2: PCA Visualization - IMPROVED ERROR MESSAGE
        with tab2:
            st.markdown('<p class="sub-header">📊 PCA Visualization</p>', unsafe_allow_html=True)
            
            if 'method' in locals() and 'sample_size' in locals():
                # Look for the specific PCA visualization for this clustering result
                pca_image_path = f"{clustering_results_path}/pca_scatter_{method}_{sample_size}.png"
                
                if os.path.exists(pca_image_path):
                    st.image(pca_image_path, caption=f"PCA Cluster Visualization ({get_display_name(selected_file)})")
                    st.markdown("Principal Component Analysis showing cluster distribution in 2D space", unsafe_allow_html=True)
                else:
                    # Improved error message
                    st.warning(f"PCA visualization not found for this {method} clustering with {sample_size} samples.")
                    st.info("Please run a new analysis with these parameters to generate the visualization.")

        # Tab 3: Dendrogram - IMPROVED ERROR MESSAGE
        with tab3:
            st.markdown('<p class="sub-header">🌲 Hierarchical Clustering Dendrogram</p>', unsafe_allow_html=True)
            
            if 'method' in locals() and 'sample_size' in locals():
                # Look for the specific dendrogram for this clustering result
                dendrogram_path = f"{clustering_results_path}/dendrogram_{method}_{sample_size}.png"
                
                if os.path.exists(dendrogram_path):
                    st.image(dendrogram_path, caption=f"Hierarchical Clustering Dendrogram ({get_display_name(selected_file)})")
                    st.markdown("Hierarchical representation of cluster relationships and distances", unsafe_allow_html=True)
                else:
                    # Improved error message
                    st.warning(f"Dendrogram visualization not found for this {method} clustering with {sample_size} samples.")
                    st.info("Please run a new analysis with these parameters to generate the visualization.")

        # Tab 4: Similarity Heatmap - IMPROVED ERROR MESSAGE
        with tab4:
            st.markdown('<p class="sub-header">🔥 Similarity Heatmap</p>', unsafe_allow_html=True)
            
            if 'method' in locals() and 'sample_size' in locals():
                # Look for the specific heatmap for this clustering result
                heatmap_path = f"{clustering_results_path}/similarity_heatmap_{method}_{sample_size}.png"
                
                if os.path.exists(heatmap_path):
                    st.image(heatmap_path, caption=f"Sample Similarity Heatmap ({get_display_name(selected_file)})")
                    st.markdown("Visualizing the similarity between document samples", unsafe_allow_html=True)
                else:
                    # Improved error message
                    st.warning(f"Similarity heatmap not found for this {method} clustering with {sample_size} samples.")
                    st.info("Please run a new analysis with these parameters to generate the visualization.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Research Paper Clustering Dashboard • Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()