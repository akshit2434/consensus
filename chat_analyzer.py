import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from google import genai
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import re

def load_mock_data(file_path):
    """Load mock chat data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [msg['message'] for msg in data['stream_chat_data']]

def light_cleanup(message):
    """Minimal cleanup while preserving original message content"""
    # Only remove excessive whitespace and normalize case
    cleaned = re.sub(r'\s+', ' ', message.strip())
    return cleaned

def generate_embeddings(messages, force_regenerate=False):
    """Generate embeddings using SentenceTransformer with caching - using original messages"""
    embeddings_path = Path("embeddings_cache.pkl")
    messages_hash_path = Path("messages_hash.pkl")
    
    # Only light cleanup, preserve original content
    cleaned_messages = [light_cleanup(msg) for msg in messages]
    
    # Create hash of current messages
    messages_hash = hash(str(sorted(cleaned_messages)))
    
    # Check if cached embeddings exist and match current messages
    if not force_regenerate and embeddings_path.exists() and messages_hash_path.exists():
        try:
            with open(messages_hash_path, 'rb') as f:
                cached_hash = pickle.load(f)
            
            if cached_hash == messages_hash:
                print("📋 Using cached embeddings...")
                with open(embeddings_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['embeddings'], cached_data['model']
        except Exception as e:
            print(f"Cache read error: {e}")
    
    # Generate new embeddings using original messages (minimal preprocessing)
    print("🔄 Generating new embeddings...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(cleaned_messages)
    
    # Cache embeddings and model
    try:
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'model': model
            }, f)
        with open(messages_hash_path, 'wb') as f:
            pickle.dump(messages_hash, f)
        print("💾 Embeddings cached for future use")
    except Exception as e:
        print(f"Cache write error: {e}")
    
    return embeddings, model

def store_in_chromadb(messages, embeddings):
    """Store messages and embeddings in ChromaDB"""
    client = chromadb.Client()
    
    # Delete collection if it exists and create new one
    try:
        client.delete_collection(name="chat_messages")
    except:
        pass
    
    collection = client.create_collection(name="chat_messages")
    
    # Add documents to collection
    ids = [f"msg_{i}" for i in range(len(messages))]
    collection.add(
        documents=messages,
        embeddings=embeddings.tolist(),
        ids=ids
    )
    
    return collection

def perform_clustering(embeddings):
    """Perform HDBSCAN clustering with high-quality dense clusters"""
    # Use configurable thresholds for meaningful clusters
    min_cluster_size = int(os.getenv('MIN_CLUSTER_SIZE', '3'))
    min_samples = int(os.getenv('MIN_SAMPLES', '2'))
    
    print(f"🎯 Clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cosine',
        cluster_selection_epsilon=0.1,  # Small epsilon for better separation
        alpha=0.8  # Balanced cluster selection
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # If no clusters found, try with slightly more relaxed parameters
    unique_clusters = set(cluster_labels)
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    if len(valid_clusters) == 0:
        print("⚠️  No clusters found, trying more relaxed parameters...")
        clusterer = HDBSCAN(
            min_cluster_size=max(2, min_cluster_size-1),
            min_samples=max(1, min_samples-1),
            metric='cosine',
            cluster_selection_epsilon=0.2,
            alpha=0.6
        )
        cluster_labels = clusterer.fit_predict(embeddings)
    
    return cluster_labels, clusterer

def get_representative_messages(embeddings, messages, cluster_labels, cluster_id):
    """Get representative messages for a cluster (closest to centroid)"""
    cluster_mask = cluster_labels == cluster_id
    cluster_embeddings = embeddings[cluster_mask]
    cluster_messages = [messages[i] for i in range(len(messages)) if cluster_mask[i]]
    
    # Calculate centroid
    centroid = np.mean(cluster_embeddings, axis=0)
    
    # Find distances to centroid
    distances = cosine_distances([centroid], cluster_embeddings)[0]
    
    # Get indices of 2-3 closest messages
    n_representatives = min(3, len(cluster_messages))
    closest_indices = np.argsort(distances)[:n_representatives]
    
    representative_messages = [cluster_messages[i] for i in closest_indices]
    return representative_messages

def extract_topic_from_cluster(representative_messages, cluster_size):
    """Extract topic from messages using frequency analysis of original content"""
    # Combine all messages in cluster (using original messages now)
    all_text = ' '.join(representative_messages).lower()
    
    # Check for neutral/indecisive patterns
    neutral_patterns = ['both', 'either', 'cant choose', 'indecisive', 'dont care', 'idc', 'undecided', 'not sure']
    if any(pattern in all_text for pattern in neutral_patterns):
        return 'Neutral'
    
    # Check for streaming meta discussion
    streaming_patterns = ['stream', 'streamer', 'chat', 'viewer', 'votes', 'voting']
    if any(pattern in all_text for pattern in streaming_patterns):
        return 'Streaming Discussion'
    
    # Extract most frequent meaningful words (not common stopwords)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    # Filter out stopwords and count frequencies
    meaningful_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    if not meaningful_words:
        return 'Other'
    
    # Count word frequencies
    word_freq = {}
    for word in meaningful_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most frequent word(s)
    if word_freq:
        # Get top 2 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:2] if freq >= 2]
        
        if top_words:
            # Capitalize and return the most relevant topic
            topic = ' '.join(top_words).title()
            return topic if len(topic) <= 20 else top_words[0].title()
    
    # Fallback to most frequent single word
    if meaningful_words:
        most_common = max(set(meaningful_words), key=meaningful_words.count)
        return most_common.title()
    
    return 'Other'

def summarize_with_gemini(representative_messages, cluster_size, api_key):
    """Use Gemini to generate topic summary for cluster or fallback to keyword extraction"""
    
    # First try keyword-based extraction
    keyword_result = extract_topic_from_cluster(representative_messages, cluster_size)
    
    if not api_key:
        print("Warning: No Gemini API key provided. Using keyword-based summaries.")
        return keyword_result
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are analyzing chat messages for a poll likely deciding between a few main options.
        Representative messages from a cluster of {cluster_size} similar chats:
        {chr(10).join(f'- "{msg}"' for msg in representative_messages)}

        What is the primary distinct opinion or choice expressed by this cluster?
        If it's about a specific game, product, or option, name it (e.g., "Jupyter", "Collab", "Option A").
        If it's expressing neutrality or indecisiveness between known options, return "Neutral/Indecisive".
        If it's a meta-comment about the stream/poll itself, return "Stream Discussion".
        If it's clearly off-topic or spam, return "Off-topic/Other".

        Return a concise topic name (1-3 words):
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt
        )
        ai_result = response.text.strip().strip('"')
        
        # Clean up AI result - should be short and meaningful
        if ai_result and len(ai_result) <= 25 and not any(char in ai_result for char in ['\n', '.', '!', '?']):
            return ai_result
        else:
            print(f"AI returned unclear result '{ai_result}', using keyword result")
            return keyword_result
            
    except Exception as e:
        print(f"Error with Gemini API: {e}, using keyword result")
        return keyword_result

def calculate_percentages(cluster_labels, total_messages):
    """Calculate percentages for each cluster"""
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    percentages = {}
    
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_messages) * 100
        percentages[label] = {
            'count': count,
            'percentage': round(percentage, 1)
        }
    
    return percentages

def main():
    print("💬 Stream Chat Analyzer - Topic-wise Poll Generator")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Phase 1: Setup & Data Prep
    print("📝 Loading mock chat data...")
    messages = load_mock_data('mock_stream_chat.json')
    print(f"Loaded {len(messages)} messages")
    
    # Phase 2: Core Logic - Embeddings & Clustering
    print("\n🔢 Generating embeddings...")
    embeddings, model = generate_embeddings(messages, force_regenerate=False)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    print("\n💾 Storing in ChromaDB...")
    collection = store_in_chromadb(messages, embeddings)
    print("Data stored in ChromaDB collection")
    
    print("\n🎯 Performing clustering...")
    cluster_labels, clusterer = perform_clustering(embeddings)
    unique_clusters = set(cluster_labels)
    outliers_count = sum(1 for label in cluster_labels if label == -1)
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    print(f"Found {len(valid_clusters)} clusters and {outliers_count} outliers")
    
    # Phase 3: Cluster Analysis & LLM Integration
    print("\n🤖 Analyzing clusters and generating summaries...")
    
    # Get environment variables
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    min_cluster_threshold = int(os.getenv('MIN_CLUSTER_THRESHOLD', '10'))
    
    if not api_key:
        print("💡 Tip: Set GEMINI_API_KEY environment variable for AI-generated summaries")
    
    cluster_summaries = {}
    small_clusters_count = 0
    small_clusters_messages = 0
    
    for cluster_id in valid_clusters:
        cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
        
        # Check if cluster meets minimum threshold
        if cluster_size < min_cluster_threshold:
            print(f"📉 Cluster {cluster_id} has only {cluster_size} messages (< {min_cluster_threshold}), moving to Others")
            small_clusters_count += 1
            small_clusters_messages += cluster_size
            continue
            
        representative_messages = get_representative_messages(
            embeddings, messages, cluster_labels, cluster_id
        )
        
        summary = summarize_with_gemini(representative_messages, cluster_size, api_key)
        cluster_summaries[cluster_id] = {
            'summary': summary,
            'size': cluster_size,
            'representative_messages': representative_messages
        }
    
    # Update outliers count to include small clusters
    total_others = outliers_count + small_clusters_messages
    print(f"📊 Small clusters moved to Others: {small_clusters_count} clusters ({small_clusters_messages} messages)")
    
    # Phase 4: Final Output
    print("\n📊 Calculating poll percentages...")
    percentages = calculate_percentages(cluster_labels, len(messages))
    
    print("\n" + "=" * 60)
    print("🗳️  GENERATED POLL OPTIONS AND RESULTS")
    print("=" * 60)
    
    # Sort clusters by size (descending) - only those that made it to cluster_summaries
    valid_cluster_ids = list(cluster_summaries.keys())
    sorted_clusters = sorted(valid_cluster_ids,
                           key=lambda x: cluster_summaries[x]['size'],
                           reverse=True)
    
    poll_results = []
    
    for i, cluster_id in enumerate(sorted_clusters, 1):
        summary = cluster_summaries[cluster_id]
        percentage_data = percentages[cluster_id]
        
        poll_option = {
            'rank': i,
            'option': summary['summary'],
            'percentage': percentage_data['percentage'],
            'vote_count': percentage_data['count'],
            'sample_messages': summary['representative_messages']
        }
        poll_results.append(poll_option)
        
        print(f"\n{i}. {summary['summary']}")
        print(f"   📊 {percentage_data['percentage']}% ({percentage_data['count']} votes)")
        print(f"   💬 Sample messages:")
        for msg in summary['representative_messages']:
            print(f"      • \"{msg}\"")
    
    # Add miscellaneous category for outliers and small clusters
    if total_others > 0:
        others_percentage = (total_others / len(messages)) * 100
        print(f"\n{len(sorted_clusters) + 1}. Other")
        print(f"   📊 {others_percentage:.1f}% ({total_others} votes)")
        print(f"   💬 Outliers + clusters < {min_cluster_threshold} messages")
        
        poll_results.append({
            'rank': len(sorted_clusters) + 1,
            'option': 'Other',
            'percentage': round(others_percentage, 1),
            'vote_count': total_others,
            'sample_messages': []
        })
    
    print("\n" + "=" * 60)
    print(f"📈 Total messages analyzed: {len(messages)}")
    print(f"🎯 Valid clusters (≥{min_cluster_threshold} messages): {len(cluster_summaries)}")
    print(f"🔍 Others (outliers + small clusters): {total_others}")
    if small_clusters_count > 0:
        print(f"📉 Small clusters moved to Others: {small_clusters_count}")
    print("=" * 60)
    
    return poll_results

if __name__ == "__main__":
    results = main()