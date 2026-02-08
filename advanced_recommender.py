
import pandas as pd
import numpy as np
import pickle
import os
import difflib
from sklearn.neighbors import NearestNeighbors

# Configuration
MODEL_PATH = 'models/recommender_model.pkl'

class AdvancedRecommender:
    def __init__(self, model_path=MODEL_PATH):
        self.load_model(model_path)
        
    def load_model(self, path):
        """Loads the pre-trained model components."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Run save_model.py first.")
            
        print(f"Loading model from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.df = data['df']
        self.tfidf = data['tfidf_vectorizer']
        self.tfidf_matrix = data['tfidf_matrix']
        self.nn_model = data['nn_model']
        print("Model loaded successfully.")

    def find_game_idx(self, title):
        """Fuzzy search for a game title."""
        title = str(title).lower()
        # 1. Exact loop
        matches = self.df[self.df['title'].str.lower() == title]
        if not matches.empty:
            return matches.index[0]
            
        # 2. Substring
        matches = self.df[self.df['title'].str.lower().str.contains(title, regex=False)]
        if not matches.empty:
            return matches.index[0]
            
        # 3. Fuzzy
        titles = self.df['title'].astype(str).tolist()
        close = difflib.get_close_matches(title, titles, n=1, cutoff=0.6)
        if close:
            return self.df[self.df['title'] == close[0]].index[0]
            
        return None

    def get_user_profile(self, game_titles):
        """
        Creates a 'User Profile Vector' by averaging the vectors of played games.
        """
        valid_indices = []
        for t in game_titles:
            idx = self.find_game_idx(t)
            if idx is not None:
                valid_indices.append(idx)
                print(f"Mapped: '{t}' -> '{self.df.iloc[idx]['title']}'")
            else:
                print(f"Skipped: '{t}' (Not found)")
                
        if not valid_indices:
            return None, []
            
        # Get vectors for played games
        user_vectors = self.tfidf_matrix[valid_indices]
        
        # Calculate centroids (User Profile)
        user_profile = np.asarray(np.mean(user_vectors, axis=0))
        
        return user_profile, valid_indices

    def recommend(self, user_games, top_k=20):
        print(f"\nGenerating recommendations based on {len(user_games)} games...")
        
        user_profile, played_indices = self.get_user_profile(user_games)
        if user_profile is None:
            return pd.DataFrame()
            
        # efficient query
        n_neighbors = 3000
        distances, indices = self.nn_model.kneighbors(user_profile, n_neighbors=n_neighbors)
        
        # Flatten
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        
        recommendations = []
        count = 0
        
        for idx, dist in zip(neighbor_indices, neighbor_distances):
            if idx in played_indices:
                continue
            
            similarity = 1 - dist
            row = self.df.iloc[idx]
            
            if row['review_count'] < 500:
                continue
                
            quality = row['norm_score']
            
            if similarity > 0.5: # Higher threshold for boost
                similarity *= 1.1
            
            # Heavy Quality Bias (0.2 Sim, 0.8 Quality)
            # This turns the recommender into "Best Rated Games within the Genre" 
            # rather than "Most Textually Similar Games"
            final_score = (similarity * 0.2) + (quality * 0.8) 
            
            recommendations.append({
                'Title': row['title'],
                'Similarity': similarity,
                'Quality': quality,
                'FinalScore': final_score,
                'Genres': row['genres'],
                'Tags': row['tags']
            })
            
            # CRITICAL: Do NOT break early! 
            # We need to evaluate ALL 3000 neighbors to let Quality Bias re-rank them.
            # If we break here, we only get the efficient "Most Similar" ones, 
            # ignoring the "Slightly less similar but 10x better" games.
                
        rec_df = pd.DataFrame(recommendations)
        if not rec_df.empty:
            rec_df = rec_df.drop_duplicates(subset='Title')
            rec_df = rec_df.sort_values('FinalScore', ascending=False).head(top_k)
        
        return rec_df

if __name__ == "__main__":
    my_games = [
        "ELDEN RING", 
        "Baldur's Gate 3", 
        "Cyberpunk 2077", 
        "The Witcher 3: Wild Hunt", 
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
    
    recommender = AdvancedRecommender()
    recs = recommender.recommend(gio_games, top_k=5)
    
    print("\n=== AI CURATED RECOMMENDATIONS (Optimization: NearestNeighbors) ===")
    if not recs.empty:
        print(recs[['Title', 'Similarity', 'FinalScore']].to_string(index=False))
