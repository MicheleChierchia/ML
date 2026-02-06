import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load Model
try:
    with open('models/steam_recommender.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['dataframe']
    tfidf_matrix = data['matrix']
except FileNotFoundError:
    print("Error: Model file 'models/steam_recommender.pkl' not found.")
    exit()

def recommend_from_played(played_games: list[str], n: int = 5) -> pd.DataFrame:
    played_indices = []
    
    for game_title in played_games:
        matches = df[df['title'].str.lower().str.contains(game_title.lower(), na=False)]
        if len(matches) > 0:
            played_indices.append(matches.index[0])
        else:
            print(f"Warning: '{game_title}' not found in dataset.")
    
    if not played_indices:
        return None
    
    # Calculate User Profile
    user_profile = np.asarray(tfidf_matrix[played_indices].mean(axis=0))
    
    # Calculate Similarity
    similarities = cosine_similarity(user_profile, tfidf_matrix)[0]
    
    # Exclude played games
    for idx in played_indices:
        similarities[idx] = -1.0
    
    # Get Top N
    top_indices = similarities.argsort()[::-1][:n]
    
    results = df.iloc[top_indices][['title']].copy()
    
    results['similarity'] = similarities[top_indices]
    
    return results

if __name__ == "__main__":
    played = ["ELDEN RING" , 
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
    "Batman: Arkham Knight",
    "STAR WARS Jedi: Survivor™",
    "STAR WARS Jedi: Fallen Order™",
    "Sekiro™: Shadows Die Twice - GOTY Edition",
    "Marvel’s Spider-Man Remastered",
    "Dead Space (2008)",
    "Sifu" ]
    
    recs = recommend_from_played(played, n=5)
    
    if recs is not None:
        # Stampa output tabellare standard di pandas
        print(recs[['title', 'similarity']].to_string(index=False))