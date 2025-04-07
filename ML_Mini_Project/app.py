import streamlit as st
import pickle
import pandas as pd
import requests
from pyngrok import ngrok

def fetch_poster(movie_id):
    try:
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3c1e3b7815ff1af8cdcf5cb5f3fc615d&language=en-US'.format(movie_id))
        data = response.json()
        return "http://image.tmdb.org/t/p/w500" + data['poster_path']
    except Exception as e:
        st.error(f"Error fetching poster: {e}")
        return None

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        recommended_movies = []
        recommended_movies_posters = []
        for i in movie_list:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_movies_posters.append(fetch_poster(movie_id))
        return recommended_movies, recommended_movies_posters
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], []

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# App UI
st.title('Movie Recommender System')
selected_movie_name = st.selectbox('Pick a Movie', movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    if names:  # Only show if we got recommendations
        cols = st.columns(5)
        for i, (name, poster) in enumerate(zip(names, posters)):
            with cols[i]:
                st.text(name)
                if poster:  # Only show image if we got the poster
                    st.image(poster, use_column_width=True)