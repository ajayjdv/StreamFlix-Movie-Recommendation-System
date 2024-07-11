import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib
from bs4 import BeautifulSoup




# Load the movie dataset
movies_df = pd.read_csv('main_data.csv')

# Load the sentiment analysis model and TfidfVectorizer
model = pickle.load(open('nlp_model.pkl', 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list



# Function to get movie details from TMDb API
#def get_movie_details(movie_title):
    #api_key = '5492165c61b1a21c06eb3a3b578a6339' # Enter your TMDb API key here
    #search_url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}&language=en-US'
    #response = requests.get(search_url)
    #if response.status_code == 200:
        #search_results = json.loads(response.content)['results']
        #if len(search_results) > 0:
            #movie_id = search_results[0]['id']
            #movie_url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
            #response = requests.get(movie_url)
            #if response.status_code == 200:
               # data = json.loads(response.content)
               # poster_path = data['poster_path']
               # title = data['original_title']
              #  overview = data['overview']
              #  genres = [genre['name'] for genre in data['genres']]
              #  return {'poster_path': f'https://image.tmdb.org/t/p/w500/{poster_path}', 'title': title, 'overview': overview, 'genres': genres}
           # else:
               # return None
        #else:
         #   return None
    #else:
       # return None

def get_movie_details(movie_title):
    api_key = '5492165c61b1a21c06eb3a3b578a6339' # Enter your TMDb API key here
    search_url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}&language=en-US'
    response = requests.get(search_url)
    if response.status_code == 200:
        search_results = json.loads(response.content)['results']
        if len(search_results) > 0:
            movie_id = search_results[0]['id']
            movie_url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
            response = requests.get(movie_url)
            if response.status_code == 200:
                data = json.loads(response.content)
                poster_path = data['poster_path']
                title = data['original_title']
                overview = data['overview']
                genres = [genre['name'] for genre in data['genres']]
                credits_url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=en-US'
                response = requests.get(credits_url)
                if response.status_code == 200:
                    credits = json.loads(response.content)
                    cast = credits['cast']
                    cast_images = []
                    for i in range(min(5, len(cast))):
                        cast_images.append(f"https://image.tmdb.org/t/p/w500/{cast[i]['profile_path']}")
                    return {'poster_path': f'https://image.tmdb.org/t/p/w500/{poster_path}', 'title': title, 'overview': overview, 'genres': genres, 'cast_images': cast_images}
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


# Function to get recommended movies based on the entered movie
# Function to get recommended movies based on the entered movie
def get_recommendations(movie_title):
    # Get the index of the entered movie
    movie_index = movies_df[movies_df['movie_title'] == movie_title].index[0]
    # Get the combination of all columns for the entered movie
    movie_combination = movies_df.iloc[movie_index]['comb']
    # Calculate the cosine similarity between the entered movie and all other movies
    cosine_similarities = np.dot(vectorizer.transform([movie_combination]), vectorizer.transform(movies_df['comb']).T).toarray()[0]
    # Get the indices of the top 5 most similar movies
    similar_indices = cosine_similarities.argsort()[::-1][:9]
    # Get the movie titles and poster paths of the top 5 most similar movies
    recommended_movies = [{'title': movies_df.iloc[index]['movie_title'], 'poster_path': get_movie_details(movies_df.iloc[index]['movie_title'])['poster_path']} for index in similar_indices]
    return recommended_movies



# Main function to run the app
def run_app():
    st.set_page_config(page_title='StreamFlix', page_icon =" :clapper: " , layout='wide')
    st.title('StreamFlix')
    # Get user input for the movie name
    movie_title = st.selectbox("Enter a movie you loved: ", movies_df['movie_title'].unique())


    # If the user has entered a movie name
    if movie_title:
        # Get the movie details from the TMDb API
        movie_details = get_movie_details(movie_title)
        if movie_details is not None:


            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(movie_details['poster_path'],use_column_width=True)

            with col2:

                st.header(movie_details['title'])
                st.subheader('Overview')
                st.write(movie_details['overview'])
                st.subheader('Genres')
                st.write(', '.join(movie_details['genres']))


                # Get the cast details from the movie dataset
            Actor_1_name = movies_df[movies_df['movie_title'] == movie_title]['actor_1_name'].values[0]
            Actor_2_name = movies_df[movies_df['movie_title'] == movie_title]['actor_2_name'].values[0]
            Actor_3_name = movies_df[movies_df['movie_title'] == movie_title]['actor_3_name'].values[0]

                # Display the cast details
                # Display the cast details
            st.subheader('Cast')
            st.write(f'{Actor_1_name}, {Actor_2_name}, {Actor_3_name}')

            col3,col4,col5 = st.columns([3,4,5])
            with col3:
                actor_1_details = get_movie_details(Actor_1_name)
                if actor_1_details is not None:
                    st.image(actor_1_details['poster_path'], width=250)
            with col4:
                actor_2_details = get_movie_details(Actor_2_name)
                if actor_2_details is not None:
                    st.image(actor_2_details['poster_path'], width=250)
            with col5:
                actor_3_details = get_movie_details(Actor_3_name)
                if actor_3_details is not None:
                    st.image(actor_3_details['poster_path'], width=250)

            # Get the movie review and sentiment

            #movie_review_tfidf = vectorizer.transform(reviews.txt)
            #movie_review_sentiment = 'Positive' if model.predict(movie_review_tfidf)[0] == 1 else 'Negative'
            # Display the movie review and sentiment
            #st.subheader('Movie Review')
            #st.write(movie_review)
            #st.subheader('Sentiment')
            #st.write(movie_review_sentiment)

            # Get recommended movies

            # Get recommended movies
            recommended_movies = get_recommendations(movie_title)
            if recommended_movies:
                # Display the recommended movies
                st.subheader('Recommended Movies')
                num_cols = 3
                col_width = int(12 / num_cols)
                cols = st.columns(num_cols)
                for i, movie in enumerate(recommended_movies):
                    with cols[i % num_cols]:
                        st.image(movie['poster_path'], width=300)
                        st.write(movie['title'] )

run_app()



