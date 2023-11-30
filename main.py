import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

file_path_ratings = 'dataset_avaliacoes.csv'
file_path_attributes = 'dataset.csv'

df_ratings = pd.read_csv(file_path_ratings)
df_attributes = pd.read_csv(file_path_attributes)

pivot_table = df_ratings.pivot_table(index='Filme', columns='Usuário', values='Avaliação').fillna(0)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(pivot_table)
similarity_matrix_ratings = euclidean_distances(normalized, normalized)

st.title("Sistema de Filtragem Colaborativa")

selected_movie = st.selectbox("Selecione um filme:", df_ratings['Filme'].unique())

def get_recommendations_hybrid(movie_name, similarity_matrix_ratings, df_attributes):
    movie_index = pivot_table.index.get_loc(movie_name)
    similar_scores_ratings = similarity_matrix_ratings[movie_index]
    similar_movies_ratings = list(pivot_table.iloc[similar_scores_ratings.argsort()][:6].index)
    similar_movies_ratings.remove(movie_name)

    movie_attributes = df_attributes[df_attributes['book title'] == movie_name]
    similar_movies_attributes = df_attributes[df_attributes['book title'] != movie_name]
    similar_movies_attributes = similar_movies_attributes[similar_movies_attributes['genre'] == movie_attributes['genre'].values[0]].head(5)['book title'].tolist()

    hybrid_recommendations = set(similar_movies_ratings[:3] + similar_movies_attributes[:3])
    return list(hybrid_recommendations)

if st.button("Mostrar recomendações"):
    recommendations = get_recommendations_hybrid(selected_movie, similarity_matrix_ratings, df_attributes)
    st.write("Filmes recomendados:")
    st.dataframe(recommendations)
