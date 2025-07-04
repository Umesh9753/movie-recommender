import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np

# --- Optional: Function to fetch movie poster ---
def fetch_poster(movie_id):
    return "https://via.placeholder.com/500x750?text=No+Image"

# --- Load model and data ---
movies_dict = pickle.load(open('movie_dict_small.pkl', 'rb'))
similarity_data = np.load("similarity_small.npz")
similarity = similarity_data['similarity']

movies = pd.DataFrame(movies_dict)

# --- Recommend function ---
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# --- Streamlit UI ---
st.title('🎬 Movie Recommender System (Top 500)')

select_movie_name = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(select_movie_name)
    for i in recommendations:
        st.write(i)
