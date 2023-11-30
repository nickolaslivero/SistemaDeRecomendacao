import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


file_path = 'dataset_avaliacoes.csv'
df = pd.read_csv(file_path)

pivot_table = df.pivot_table(index='Filme', columns='Usuário', values='Avaliação').fillna(0)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(pivot_table)
similarity_matrix = euclidean_distances(normalized, normalized)

st.title("Sistema de Filtragem Colaborativa")

selected_movie = st.selectbox("Selecione um filme:", df['Filme'].unique())

def get_recommendations_euclidean(movie_name, similarity_matrix):
    movie_index = pivot_table.index.get_loc(movie_name)
    similar_scores = similarity_matrix[movie_index]
    similar_movies = list(pivot_table.iloc[similar_scores.argsort()][:6].index)
    similar_movies.remove(movie_name)
    return similar_movies[:5]

if st.button("Mostrar recomendações"):
    recommendations = get_recommendations_euclidean(selected_movie, similarity_matrix)
    st.write("Filmes recomendados:")
    st.dataframe(recommendations)
