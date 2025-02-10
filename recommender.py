import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_based_filtering():
   
    item_file = "ml-100k/u.item" 
    columns = ['movie_id', 'title'] + [f'genre_{i}' for i in range(22)] 
    movies_df = pd.read_csv(item_file, sep='|', names=columns, encoding='latin-1')
    
    
    movies_df['genres'] = movies_df.iloc[:, 2:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
    
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return movies_df, cosine_sim

def get_movie_recommendations(movie_title, movies_df, cosine_sim, top_n=5):
   
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
   
    sim_scores = list(enumerate(cosine_sim[idx]))
    
   
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
   
    sim_scores = sim_scores[1:top_n + 1]  
    
    # Get the movie titles based on similarity scores
    recommended_movies = [movies_df['title'][i[0]] for i in sim_scores]
    
    return recommended_movies
