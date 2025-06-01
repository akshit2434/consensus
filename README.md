# Stream Chat Analyzer - Topic-wise Poll Generator

A generic Python script that processes any stream chat messages to generate topic-wise poll options and percentages using machine learning clustering and AI summarization. Works for any type of content - gaming, tech discussions, product comparisons, etc.

## Features

- **Embedding Generation**: Uses SentenceTransformer to convert chat messages into vector embeddings
- **Persistent Caching**: Stores embeddings locally to avoid regeneration on subsequent runs
- **Smart Clustering**: Uses optimized HDBSCAN parameters to minimize outliers and maximize meaningful clusters
- **Visual Analysis**: Includes visualization script with PCA/t-SNE plots for development and testing
- **AI Summarization**: Uses Google Gemini API to generate meaningful poll options
- **Statistical Analysis**: Calculates percentages and provides representative samples

## Setup

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional - Set Gemini API Key** (for AI-generated summaries):
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Analysis
Run the main analyzer with your mock chat data:

```bash
python chat_analyzer.py
```

### Visualization and Development
Run the visualization script to see clusters and test parameters:

```bash
python visualize_clusters.py
```

### Workflow
1. **First run**: `python chat_analyzer.py` - generates and caches embeddings
2. **Subsequent runs**: Uses cached embeddings for faster processing
3. **Visualization**: `python visualize_clusters.py` - creates PCA/t-SNE plots and detailed analysis
4. **Force regeneration**: Modify `force_regenerate=True` in the script if needed

The scripts will:
1. Load chat messages from `mock_stream_chat.json`
2. Generate/load cached embeddings using SentenceTransformer
3. Store data in ChromaDB
4. Perform optimized clustering to minimize outliers
5. Generate poll options (with AI if API key is provided)
6. Display results with percentages and sample messages
7. (Visualization) Create cluster plots and save as PNG files

## Output Example

### Main Analyzer Results (Generic Topics)
```
🗳️  GENERATED POLL OPTIONS AND RESULTS
============================================================

1. Roblox
   📊 23.5% (12 votes)
   💬 Sample messages:
      • "roblox has more variety of games tbh"
      • "roblox is better tho ngl"

2. Minecraft
   📊 13.7% (7 votes)
   💬 Sample messages:
      • "MINECRAFT!!! pls pls pls"
      • "minecraft creative mode hits different"

3. Streaming Discussion
   📊 11.7% (6 votes)
   💬 Sample messages:
      • "guys chill lets just see what streamer wants"
      • "idc what we play just start streaming"

4. Game Comparison
   📊 5.9% (3 votes)
   💬 Sample messages:
      • "cant choose they both good 😭"
      • "why not both? 🤔"

5. Other
   📊 41.2% (21 votes)
   💬 Uncategorized messages
```

### Visualization Output
- **PCA Plot**: `cluster_visualization_pca.png` - Principal component analysis view
- **t-SNE Plot**: `cluster_visualization_tsne.png` - Non-linear dimensionality reduction view
- **Parameter Comparison**: Tests different clustering settings
- **Detailed Analysis**: Complete breakdown of all clusters and outliers
```

## Configuration

- **Clustering Parameters**: Adjust `min_cluster_size` in `perform_clustering()` function
- **Model Selection**: Change the SentenceTransformer model in `generate_embeddings()` (currently using 'all-mpnet-base-v2')
- **Gemini Model**: Update the model name in `summarize_with_gemini()` function
- **Content Agnostic**: Works for any type of stream content - gaming, tech, product discussions, etc.

## Dependencies

- `sentence-transformers`: For generating text embeddings
- `chromadb`: For vector storage and retrieval
- `hdbscan`: For density-based clustering
- `google-genai`: For AI-powered summarization
- `numpy`: For numerical operations
- `scikit-learn`: For machine learning utilities
- `matplotlib`: For creating visualizations
- `seaborn`: For enhanced plot styling

## Key Improvements

- **Reduced Outliers**: Improved from 78.4% to 9.8% outliers through optimized clustering
- **Persistent Caching**: Embeddings are cached locally to speed up subsequent runs
- **Visual Development**: Comprehensive visualization script for parameter tuning and analysis
- **Better Clustering**: Multiple parameter testing to find optimal cluster configurations

## Files

- `chat_analyzer.py`: Main analysis script
- `visualize_clusters.py`: Visualization and development tool
- `requirements.txt`: Dependencies list
- `mock_stream_chat.json`: Sample chat data
- `embeddings_cache.pkl`: Cached embeddings (generated automatically)
- `messages_hash.pkl`: Message hash for cache validation (generated automatically)
- `cluster_visualization_*.png`: Generated visualization plots

## Notes

- The script works without a Gemini API key (uses placeholder summaries)
- Embeddings are automatically cached for faster subsequent runs
- Designed as a test script - minimal error handling for simplicity
- Ready to be extended into a microservice/worker architecture
- Use visualization script for development and parameter tuning