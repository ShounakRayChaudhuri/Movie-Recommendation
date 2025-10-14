import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ✅ Securely load OMDb API key (no hardcoding)
OMDB_API_KEY = None
if "OMDB_API_KEY" in st.secrets:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
else:
    OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if not OMDB_API_KEY:
    st.error("❌ OMDb API key not found. Please set it in Streamlit secrets or environment variables.")

# Function to fetch movie info from OMDb API
@st.cache_data
def fetch_movie_info(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if data['Response'] == 'True':
        return {
            'poster': data.get('Poster'),
            'plot': data.get('Plot'),
            'year': data.get('Year'),
            'genre': data.get('Genre'),
            'director': data.get('Director'),
            'actors': data.get('Actors')
        }
    else:
        return None

# Load datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select relevant features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Function to convert stringified lists into Python lists
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Apply conversion
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

# Create tags
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# Final dataframe
final_movies = movies[['movie_id', 'title', 'tags']]

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(final_movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = final_movies[final_movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10
    movie_indices = [i[0] for i in sim_scores]
    return final_movies['title'].iloc[movie_indices]

# Streamlit UI
st.title('🎬 Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie:',
    final_movies['title'].values
)

if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie)
    st.write('Top 10 movie recommendations:')
    
    for movie in recommendations:
        info = fetch_movie_info(movie)
        if info:
            col1, col2 = st.columns([1, 3])
            with col1:
                if info['poster'] and info['poster'] != 'N/A':
                    st.image(info['poster'], width=150)
                else:
                    st.write("Poster not available")
            with col2:
                st.subheader(movie)
                st.markdown(f"**Year:** {info['year']}")
                st.markdown(f"**Genre:** {info['genre']}")
                st.markdown(f"**Director:** {info['director']}")
                st.markdown(f"**Actors:** {info['actors']}")
                st.markdown(f"**Plot:** {info['plot']}")
        else:
            st.warning(f"No data found for {movie}")
