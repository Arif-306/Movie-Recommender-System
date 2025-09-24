# movie_recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
movies = {
    'title': [
        'The Matrix', 'The Godfather', 'The Dark Knight',
        'Pulp Fiction', 'Inception', 'Interstellar'
    ],
    'genre': [
        'Action Sci-Fi', 'Crime Drama', 'Action Crime',
        'Crime Drama', 'Action Sci-Fi', 'Adventure Sci-Fi'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(movies)

# Vectorize genres
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['genre'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(movie_title, n=3):
    if movie_title not in df['title'].values:
        return ["Movie not found in database."]
    
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Test
print("Recommendations for Inception:", recommend('Inception'))
