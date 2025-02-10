# app.py

import streamlit as st
from recommender import content_based_filtering, get_movie_recommendations

# Title of the Streamlit app
st.title("Movie Recommender System (Content-Based)")

# Load Content-Based Filtering Model
movies_df, cosine_sim = content_based_filtering()

# Input fields for user interaction
movie_title = st.text_input("Enter Movie Title:", "Toy Story (1995)")

if st.button("Get Recommendations"):
    # Get Content-Based recommendations
    cb_recs = get_movie_recommendations(movie_title, movies_df, cosine_sim, top_n=5)
    
    # Display results
    st.write(f"Content-Based recommendations for '{movie_title}':")
    for rec in cb_recs:
        st.write(f"- {rec}")
