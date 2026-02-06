import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import difflib

# Load Model
try:
    with open('models/steam_recommender.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['dataframe']
    tfidf_matrix = data['matrix']
except FileNotFoundError:
    print("Error: Model file 'models/steam_recommender.pkl' not found.")
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
    # This is crucial for typos like "Kinght"
    all_titles = df['title'].tolist()
    close_matches = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.5)
    if close_matches:
        best_match = close_matches[0]
        print(f"Fuzzy matching '{title}' to '{best_match}'")
        return df[df['title'] == best_match].index[0]
        
    return None

def recommend_advanced(played_games: list[str], blacklist: list[str] = None, n: int = 5):
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
    played_log_reviews = np.log1p(played_df['total_reviews'].astype(float))
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
        
        # Calculate Similarity
        user_vector = tfidf_matrix[played_indices].mean(axis=0)
        # Reshape for sklearn
        user_vector = np.asarray(user_vector).reshape(1, -1)
        
        sim_scores = cosine_similarity(user_vector, cluster_matrix)[0]
        
        # Create temp dataframe for scoring
        candidates = df.loc[cluster_indices].copy()
        candidates['similarity'] = sim_scores
        
        # 3. Apply Popularity Distance Logic
        candidates['log_reviews'] = np.log1p(candidates['total_reviews'].astype(float))
        
        # Calculate how much the candidate's popularity deviates from the user's median
        pop_distance = abs(candidates['log_reviews'] - user_target_pop)
        candidates['pop_score'] = 1 / (1 + pop_distance)
        
        # Final Score: Blend similarity with popularity alignment
        candidates['final_score'] = candidates['similarity'] * 0.7 + candidates['pop_score'] * 0.3

        # 4. Filtering
        # Exclude played games
        candidates = candidates[~candidates.index.isin(played_indices)]
        
        # Exclude blacklist (substring match to catch franchises)
        if blacklist:
            for item in blacklist:
                clean_item = item.lower().strip().replace('’', "'")
                # Remove common special characters from the dataframe titles temporarily for matching
                titles_to_check = candidates['title'].str.lower().str.replace('’', "'").str.replace('®', '').str.replace('™', '')
                candidates = candidates[~titles_to_check.str.contains(clean_item, regex=False)]
        
        # Get Top K for this cluster
        top_k = candidates.sort_values('final_score', ascending=False).head(slots)
        recommendations.append(top_k)
    
    # 5. Final Assemble
    if not recommendations:
        return pd.DataFrame()
        
    final_df = pd.concat(recommendations)
    
    # Re-sort by final score to ensure best quality
    final_df = final_df.sort_values('final_score', ascending=False).head(n)
    
    return final_df[['title', 'genres', 'total_reviews', 'cluster', 'final_score']]

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
    "Marvel’s Spider-Man Remastered",
    "Dead Space (2008)",
    "Sifu",
    "Batman Arkham Kinght",
    " Batman™: Arkham Origins"]

    
    my_blacklist = ["Assassin's Creed"]

    recs = recommend_advanced(my_games, blacklist=my_blacklist, n=5)
    
    if recs is not None:
        print("\nTop Recommendations:")
        print(recs.to_string(index=False))