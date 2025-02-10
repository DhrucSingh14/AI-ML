# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Content-Based Filtering using movie genres
def content_based_filtering():
    # Load movie data from the u.item file (MovieLens dataset)
    item_file = "ml-100k/u.item"  # Make sure the file path is correct
    columns = ['movie_id', 'title'] + [f'genre_{i}' for i in range(22)]  # 22 genres in MovieLens
    movies_df = pd.read_csv(item_file, sep='|', names=columns, encoding='latin-1')
    
    # Combine genre columns into a single text field
    movies_df['genres'] = movies_df.iloc[:, 2:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    # Use TfidfVectorizer to convert genre text into a numerical feature matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
    
    # Calculate cosine similarity between movies based on their genre features
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return movies_df, cosine_sim

# Function to get movie recommendations based on content-based filtering
def get_movie_recommendations(movie_title, movies_df, cosine_sim, top_n=5):
    # Find the index of the movie that matches the input title
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get similarity scores for all movies with the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar movies (excluding the input movie)
    sim_scores = sim_scores[1:top_n + 1]  # Skip the first one because it's the same movie
    
    # Get the movie titles based on similarity scores
    recommended_movies = [movies_df['title'][i[0]] for i in sim_scores]
    
    return recommended_movies
