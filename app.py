from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
import json
from imdb import IMDb
import time
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import matplotlib
matplotlib.use('Agg') 
import re
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, template_folder='templates')
logging.basicConfig(level=logging.INFO)

# Global data loading
prec, ratings, tagged_movies, crec, tf_sim_score, sim_score = None, None, None, None, None, None

def load_data():
    global prec, ratings, tagged_movies, crec, tf_sim_score, sim_score
    if prec is None:  # Check if data is already loaded
        prec = pd.read_csv(r'D:\Codes\Python\MyProject\moviemind\data\filtered_movies.csv')
        ratings = pd.read_csv(r'D:\Codes\Python\MyProject\moviemind\data\ratings.csv')
        tagged_movies = pd.read_csv(r'D:\Codes\Python\MyProject\moviemind\data\tags_filtered.csv')

        # Convert genre strings to lists
        prec['genres'] = prec['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        ratings = ratings[ratings['movieId'].isin(prec['movieId'])]
        rc = ratings.groupby('userId').count()['rating'].sort_values(ascending=False).reset_index()
        rc = rc[rc['rating'] >= 500]
        ratings = ratings[ratings['userId'].isin(rc['userId'])]

        crec = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        tf = TfidfVectorizer(max_features=20000, stop_words='english')
        tf_tags = tf.fit_transform(tagged_movies['tag']).toarray()
        tf_sim_score = cosine_similarity(tf_tags)
        sim_score = cosine_similarity(crec)

# Initialize data on startup
load_data()

# Helper functions for data processing
def get_movie_poster(imdb_id, fallback_url="static/images/no_poster.jpg"):
    global prec  # Ensure you have access to the global DataFrame
    
    # Check if the IMDb ID exists in the DataFrame
    movie_row = prec[prec['imdbId'] == imdb_id]
    if not movie_row.empty:
        return movie_row['image'].values[0]  # Return the image path from the DataFrame

    # If the image is not found, return the fallback URL
    return fallback_url

# Routes
@app.route('/')
def home():
    
    # Create a line plot for Movie Counts by Year
    year_graph_path = 'D:/Codes/Python/MyProject/moviemind/static/graphs/movie_counts_by_year.png'
    # Create a pie chart for Movie Counts by Genre
    genre_graph_path = 'D:/Codes/Python/MyProject/moviemind/static/graphs/movie_counts_by_genre.png'
    # Create a bar chart for Top 10 Highest Grossing Movies
    grossing_graph_path = 'D:/Codes/Python/MyProject/moviemind/static/graphs/top_10_highest_grossing_movies.png'
    # Create a bar chart for Top 10 Highest Rated Movies
    rated_graph_path = 'D:/Codes/Python/MyProject/moviemind/static/graphs/top_10_highest_rated_movies.png'
    
    # Prepare all graph paths for rendering
    graphs = [
        year_graph_path,
        genre_graph_path,
        grossing_graph_path,
        rated_graph_path
    ]

    return render_template('home.html', graphs=graphs)


@app.route('/movies', methods=['GET'])
def movies():
    global top_movies
    load_data()  # Ensure data is loaded
    top_movies = prec.copy()
    top_movies = top_movies.sort_values(by=['count', 'mean_ratings'], ascending=[False, False])
    top_movies = top_movies.head(20).to_dict(orient='records')
    print(top_movies)  # Convert DataFrame to a list of dictionaries
    return render_template('movies.html', movies=top_movies, prec=prec)

@app.route('/api/filter_movies', methods=['GET'])
def filter_movies():
    selected_years = request.args.getlist('years[]')
    selected_genres = request.args.getlist('genres[]')
    filtered_movies = prec.copy()

    # Apply filters based on user selection
    if selected_years and selected_years[0] != '':
        filtered_movies = filtered_movies[filtered_movies['year'].isin(map(int, selected_years))]
        filtered_movies = filtered_movies.sort_values(by=['count', 'mean_ratings'], ascending=[False, False])
    if selected_genres and selected_genres[0] != '':
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
        filtered_movies = filtered_movies.sort_values(by=['count', 'mean_ratings'], ascending=[False, False])
    # apply based on both for example(selecting year 2020 from years and 'Action' from genre)

    # If no filters are applied, return top 20 movies
    if filtered_movies.empty or (not selected_years and not selected_genres):
        return jsonify(top_movies)

    return jsonify(filtered_movies.to_dict(orient='records'))


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    movie_name = request.args.get('movie')
    
    logging.info(f"Received recommendation request for movie: {movie_name}")

    try:
        # Adjusted search to be more exact
        movie_data = prec[prec['name'].str.contains(movie_name, case=False) & 
                          prec['name'].str.match(rf'^{re.escape(movie_name)}$')]
        
        if movie_data.empty:
            logging.warning(f"Movie not found: {movie_name}")
            return jsonify({'error': 'Movie not found'})

        movie = movie_data.iloc[0]
        movie_id = movie['movieId']
        logging.info(f"Found movie: {movie['name']} (ID: {movie_id})")

        # Content-based filtering
        content_sims = []
        if movie_id in tagged_movies['movieId'].values:
            content_idx = tagged_movies[tagged_movies['movieId'] == movie_id].index[0]
            content_sims = list(enumerate(tf_sim_score[content_idx]))
            logging.info(f"Content-based similarities calculated for movie ID: {movie_id}")

        # Collaborative filtering
        collab_sims = []
        if movie_id in crec.index:
            collab_idx = crec.index.get_loc(movie_id)
            collab_sims = list(enumerate(sim_score[collab_idx]))
            logging.info(f"Collaborative similarities calculated for movie ID: {movie_id}")

        # Combine scores
        combined_scores = {}
        collab_weight = 0.3

        for idx, score in content_sims:
            movie_id_content = tagged_movies['movieId'].iloc[idx]
            if movie_id_content != movie_id:
                combined_scores[movie_id_content] = score * (1 - collab_weight)

        for idx, score in collab_sims:
            movie_id_collab = crec.index[idx]
            combined_scores[movie_id_collab] = combined_scores.get(movie_id_collab, 0) + score * collab_weight

        recommended_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:21]
        recommended_movies = prec[prec['movieId'].isin(recommended_ids)]
        recommended_movies = recommended_movies[recommended_movies['name'] != movie_name]
        recommended_movies['mean_ratings'] = recommended_movies['mean_ratings'].apply(lambda x: round(x, 1))
        frecommended_movies = recommended_movies.sort_values(by=['count', 'mean_ratings'], ascending=[False, False])
        # Use the 'image' column directly for recommended movies
        recommended_movies['poster'] = recommended_movies['image']

        logging.info(f"Returning {len(recommended_movies)} recommended movies")
        return jsonify(recommended_movies.to_dict(orient='records'))

    except Exception as e:
        logging.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
