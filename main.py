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

def get_recommendations_euclidean(movie_name, similarity_matrix, pivot_table):
    movie_index = pivot_table.index.get_loc(movie_name)
    similar_scores = similarity_matrix[movie_index]
    similar_movies = list(pivot_table.iloc[similar_scores.argsort()][:6].index)
    similar_movies.remove(movie_name)
    
    similar_movies_with_mean = []
    for movie in similar_movies:
        mean_rating = df[df['Filme'] == movie]['Avaliação'].mean()
        similar_movies_with_mean.append((movie, mean_rating))

    return similar_movies_with_mean[:5]

if st.button("Mostrar recomendações"):
    recommendations = get_recommendations_euclidean(selected_movie, similarity_matrix, pivot_table)
    recommendations_df = pd.DataFrame(recommendations, columns=['Filme Recomendado', 'Média de Avaliação'])
    st.write("Filmes recomendados:")
    st.dataframe(recommendations_df)

# Seção para avaliação do filme pelo usuário
st.header("Avalie um Filme")
user_movie = st.selectbox("Selecione um filme para avaliar:", df['Filme'].unique())
user_rating = st.slider("Avalie o filme:", 1, 5)
submit_button = st.button("Enviar Avaliação")

if submit_button:
    new_rating = pd.DataFrame({'Usuário': ['Novo'], 'Filme': [user_movie], 'Avaliação': [user_rating]})
    df = pd.concat([df, new_rating], ignore_index=True)
    st.success("Avaliação enviada com sucesso!")
