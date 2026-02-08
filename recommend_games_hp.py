import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load High Precision Model
# Load High Precision Model
try:
    # 1. Load original data metadata
    df_orig = pd.read_csv('datasets/steam-games-cleaned.csv')
    
    # 2. Load DBSCAN model and clusters
    with open('models/dbscan_model.pkl', 'rb') as f:
        data = pickle.load(f)
        
    df_clusters = data['df_clusters']
    
    # 3. Merge to get full dataframe
    # Assuming app_id is the key. 
    if 'app_id' in df_orig.columns and 'app_id' in df_clusters.columns:
        df = pd.merge(df_orig, df_clusters[['app_id', 'cluster']], on='app_id', how='inner')
    else:
        # Fallback to title match if app_id missing (unlikely)
        df = pd.merge(df_orig, df_clusters[['title', 'cluster']], on='title', how='inner')
        
    # 4. Use SVD-reduced scaled matrix for similarity (fast and consistent with clustering)
    # This is (N, 5) matrix
    tfidf_matrix = data['cluster_matrix'] 
    
    print(f"Loaded model: DBSCAN, Clusters: {df['cluster'].nunique()}, Noise: {(df['cluster']==-1).sum()}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def get_game_idx(title):
    title_lower = title.lower()
    # 1. Exact match (case insensitive)
    matches = df[df['title'].str.lower() == title_lower]
    if len(matches) > 0:
        return matches.index[0]
    
    # 2. Substring match
    matches = df[df['title'].str.lower().str.contains(title_lower, regex=False)]
    if len(matches) > 0:
        return matches.index[0]
        
    # 3. Fuzzy search fallback (difflib)
    all_titles = df['title'].tolist()
    close_matches = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.5)
    if close_matches:
        best_match = close_matches[0]
        print(f"Fuzzy matching '{title}' to '{best_match}'")
        return df[df['title'] == best_match].index[0]
        
    return None

def recommend_advanced(
    played_games: list[str], 
    blacklist: list[str] = None, 
    n: int = 5,
    diversity_weight: float = 0.3,
    pop_threshold: float = 3.0,
    min_review_score: float = 0.0
):

    played_indices = []
    
    print(f"Analyzing {len(played_games)} played games...")
    
    for title in played_games:
        idx = get_game_idx(title.strip())
        if idx is not None:
            played_indices.append(idx)
        else:
            print(f"Warning: '{title}' not found.")
            
    if not played_indices:
        return None

    # 1. Analyze User Profile
    played_df = df.iloc[played_indices]
    
    # Cluster Distribution (Proportions)
    cluster_counts = played_df['cluster'].value_counts(normalize=True)
    
    # Popularity Profile (Log scale to handle huge variance)
    played_log_reviews = np.log1p(played_df['review_count'].astype(float))
    user_target_pop = played_log_reviews.median() 
    
    print(f"User Profile: Target Popularity (Log Median)={user_target_pop:.2f}")
    print("Cluster Distribution:")
    print(cluster_counts.to_string())

    recommendations = []
    
    # 2. Generate Recommendations per Cluster
    for cluster_id, proportion in cluster_counts.items():
        # Calculate how many slots for this cluster
        slots = max(1, int(round(n * proportion)))
        
        # Filter dataframe for this cluster
        cluster_mask = df['cluster'] == cluster_id
        cluster_indices = df[cluster_mask].index
        
        if len(cluster_indices) == 0:
            continue
            
        # Get vectors
        cluster_matrix = tfidf_matrix[cluster_indices]
        
        # Calculate Similarity - CLUSTER SPECIFIC STRATEGY
        # Find which of the user's played games belong to THIS cluster
        # This ensures we recommend games similar to the SPECIFIC ones the user played in this genre/cluster
        user_games_in_cluster_mask = played_df['cluster'] == cluster_id
        user_indices_in_cluster = played_df[user_games_in_cluster_mask].index
        
        if len(user_indices_in_cluster) > 0:
            # Create a user vector specifically for this cluster
            # Average vector of played games ONLY in this cluster
            cluster_user_vector = tfidf_matrix[user_indices_in_cluster].mean(axis=0)
        else:
            # Fallback (should rarely happen given logic): use global profile
            cluster_user_vector = tfidf_matrix[played_indices].mean(axis=0)
            
        # Reshape for sklearn
        cluster_user_vector = np.asarray(cluster_user_vector).reshape(1, -1)
        
        sim_scores = cosine_similarity(cluster_user_vector, cluster_matrix)[0]
        
        # Create temp dataframe for scoring
        candidates = df.loc[cluster_indices].copy()
        candidates['similarity'] = sim_scores
        
        # 3. Apply Popularity Distance Logic
        candidates['log_reviews'] = np.log1p(candidates['review_count'].astype(float))
        
        # Calculate how much the candidate's popularity deviates from the user's median
        pop_distance = abs(candidates['log_reviews'] - user_target_pop)
        candidates['pop_score'] = 1 / (1 + pop_distance)
        
        # Quality Score from review_score (0-100 scale -> 0-1)
        # Using review_score column as verified in steam_recommender.pkl
        candidates['quality_score'] = candidates['review_score'].fillna(0) / 100
        
        # Final Score: Blend similarity, popularity, and quality
        # Weights: 40% Similarity, 35% Popularity fit, 25% Review Score (Modified based on user feedback)
        candidates['final_score'] = (
            candidates['similarity'] * 0.40 + 
            candidates['pop_score'] * 0.35 +
            candidates['quality_score'] * 0.25
        )

        # 4. Filtering
        # Exclude played games
        candidates = candidates[~candidates.index.isin(played_indices)]
        
        # Filtro popolarità - escludi giochi troppo diversi dal profilo utente
        pop_diff = abs(candidates['log_reviews'] - user_target_pop)
        candidates = candidates[pop_diff <= pop_threshold]
        
        # Filtro qualità - escludi giochi con recensioni troppo negative
        candidates = candidates[candidates['review_score'].fillna(0) >= min_review_score]
        
        # Exclude blacklist (substring match to catch franchises)
        if blacklist:
            for item in blacklist:
                clean_item = item.lower().strip()
                candidates = candidates[~candidates['title'].str.lower().str.contains(clean_item, regex=False, na=False)]
        
        # Get Top K for this cluster
        top_k = candidates.sort_values('final_score', ascending=False).head(slots)
        recommendations.append(top_k)
    
    # 5. Final Assemble with Diversity (MMR-style)
    if not recommendations:
        return pd.DataFrame()
        
    all_candidates = pd.concat(recommendations)
    
    # Get TF-IDF vectors for all candidates
    candidate_indices = all_candidates.index.tolist()
    candidate_vectors = tfidf_matrix[candidate_indices]
    
    # Diverse selection using Maximal Marginal Relevance (MMR)
    selected = []
    selected_indices = []
    remaining = all_candidates.copy()
    
    while len(selected) < n and len(remaining) > 0:
        if len(selected_indices) == 0:
            # First pick: just take the best score
            best_idx = remaining['final_score'].idxmax()
        else:
            # Calculate similarity to already selected games
            selected_vectors = tfidf_matrix[selected_indices]
            remaining_indices = remaining.index.tolist()
            remaining_vectors = tfidf_matrix[remaining_indices]
            
            # Similarity of each remaining candidate to all selected
            sim_to_selected = cosine_similarity(remaining_vectors, selected_vectors)
            max_sim = sim_to_selected.max(axis=1)  # Max similarity to any selected
            
            # MMR: original_score - diversity_weight * max_similarity_to_selected
            mmr_scores = remaining['final_score'].values - diversity_weight * max_sim
            
            best_pos = mmr_scores.argmax()
            best_idx = remaining_indices[best_pos]
        
        selected.append(remaining.loc[best_idx])
        selected_indices.append(best_idx)
        remaining = remaining.drop(best_idx)
    
    final_df = pd.DataFrame(selected)
    
    if final_df.empty:
        return None
        
    # Output più dettagliato
    output_cols = ['title', 'genres', 'review_score', 'review_count', 'final_score']
    # Check if 'features' column exists, if so add it
    if 'features' in final_df.columns:
        output_cols.append('features')
        
    return final_df[output_cols]


if __name__ == "__main__":
    # Test
    my_games = ["ELDEN RING" , 
    "Baldur's Gate 3" , 
    "Cyberpunk 2077" , 
    "The Witcher 3: Wild Hunt" , 
    "Fallout 3: Game of the Year Edition",
    "Lords of the Fallen",
    "Hades",
    "The Elder Scrolls V: Skyrim Special Edition",
    "DOOM Eternal",
    "DOOM",
    "Batman: Arkham City - Game of the Year Edition",
    "STAR WARS Jedi: Survivor™",
    "STAR WARS Jedi: Fallen Order™",
    "Sekiro™: Shadows Die Twice - GOTY Edition",
    "Marvel's Spider-Man Remastered",
    "Dead Space (2008)",
    "Sifu",
    "Batman: Arkham Asylum Game of the Year Edition",
    "Batman Arkham Kinght",
    " Batman™: Arkham Origins",
    "Thymesia",
    "The Last of Us Part I",
    "Wolfenstein the new order",
    "Wolfenstein the new colossus"
    ]

    gio_games = ["Baldur's Gate 3", 
    "Dying Light",
    "Dead By Daylight",
    "Sifu"]

    my_blacklist = ["Assassin's Creed"]

    print("--- Recommendations for 'my_games' ---")
    recs = recommend_advanced(gio_games, blacklist=my_blacklist, n=10)
    
    if recs is not None:
        print("\nTop Recommendations:")
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.width', 1000)
        print(recs.to_string(index=False))
