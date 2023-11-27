import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

file_path = 'dataset.csv'
df = pd.read_csv(file_path)


df['combined_features'] = df['book title'] + ' ' + df['author'] + ' ' + df['genre']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(book_title):
    idx = df[df['book title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Recomendar os 5 livros mais similares
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][['book title', 'author', 'genre', 'year of publication', 'rating']]


st.title('Sistema de Recomendação de Livros')

selected_book = st.selectbox('Selecione um livro:', df['book title'].unique())

if st.button('Obter Recomendações'):
    recommendations = get_recommendations(selected_book)
    st.table(recommendations)
