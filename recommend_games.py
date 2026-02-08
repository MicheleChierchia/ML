import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import difflib

# Load Model
try:
    with open('models/steam_recommender_hp.pkl', 'rb') as f:
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
        
        # Calculate Similarity
        user_vector = tfidf_matrix[played_indices].mean(axis=0)
        # Reshape for sklearn
        user_vector = np.asarray(user_vector).reshape(1, -1)
        
        sim_scores = cosine_similarity(user_vector, cluster_matrix)[0]
        
        # Create temp dataframe for scoring
        candidates = df.loc[cluster_indices].copy()
        candidates['similarity'] = sim_scores
        
        # 3. Apply Popularity Distance Logic
        candidates['log_reviews'] = np.log1p(candidates['review_count'].astype(float))
        
        # Calculate how much the candidate's popularity deviates from the user's median
        pop_distance = abs(candidates['log_reviews'] - user_target_pop)
        candidates['pop_score'] = 1 / (1 + pop_distance)
        
        # Final Score: Blend similarity with popularity alignment
        candidates['final_score'] = candidates['similarity'] * 0.7 + candidates['pop_score'] * 0.3

        # 4. Filtering
        # Exclude played games
        candidates = candidates[~candidates.index.isin(played_indices)]
        
        # NUOVO: Filtro popolarità - escludi giochi troppo diversi dal profilo utente
        POPULARITY_THRESHOLD = 3.0  # Max differenza in log scale (~20x in review count)
        pop_diff = abs(candidates['log_reviews'] - user_target_pop)
        candidates = candidates[pop_diff <= POPULARITY_THRESHOLD]
        
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
    
    diversity_weight = 0.3  # How much to penalize similarity to already selected
    
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
    
    return final_df[['title']]


if __name__ == "__main__":
    # Test
    my_games = [
    "A little to the left",
    "Story Tellers",
    "Unpacking",
    "Lost in random",
    "House Flipper"]

    gio_games = [
  "Dying Light",
  "Dead Island 2",
  "The Long Dark",
  "Subnautica",
  "Sifu",
  "Cyberpunk 2077",
  "Resident Evil 2",
  "Resident Evil 4",
  "State of Decay 2",
  "Far Cry 5",
  "Fallout 4",
  "Slime Rancher"
]

    dario_games =["Outer Wilds", "The Last of Us: Part II", "Terraria", "Baldur's Gate 3", "Ratchet and Clank: Rift Apart", "Elden Ring", "Mass Effect Legendary Edition", "Fallout 3", "Balatro", "The Binding of Isaac: Rebirth", "Half Life 2", "Assassin's Creed: Brotherhood", "Hollow Knight", "Sekiro: Shadow Die Twice", "God of War", "Death Stranding", "Portal 2", "Resident Evil", "Chants of Sennaar", "Ori and the Will of the Wisps", "Hades", "Monster Hunter: World", "Remnant: From the Ashes", "Remnant 2", "Dark Souls Remastered", "Dark Souls 3", "Bioshock", "Death's Door", "Beyond Good and Evil", "Detroit Become Human", "Far Cry 3", "Batman Arkham Asylum"]

    my_blacklist = []

    recs = recommend_advanced(my_games, blacklist=my_blacklist, n=10)
    
    if recs is not None:
        print("\nTop Recommendations:")
        print(recs.to_string(index=False))