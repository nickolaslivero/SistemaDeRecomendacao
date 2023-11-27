import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Carregar o dataset localmente
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Processamento dos dados (tratamento de NaN, se necessário)
# Exemplo: df['book title'].fillna('', inplace=True)

# Vetorização do texto
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['book title'])

# Cálculo da matriz de similaridade (usando similaridade do cosseno)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Função para obter recomendações de livros com base no título inserido
def get_recommendations(book_title):
    # Verificar se o título do livro está no dataset
    if book_title not in df['book title'].values:
        return "Livro não encontrado. Tente outro título."

    idx = df[df['book title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Recomendar os 5 livros mais similares
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices]['book title']

# Interface Streamlit
st.title('Sistema de Recomendação de Livros')

book_input = st.text_input('Digite o nome de um livro:', 'The Great Gatsby')  # Exemplo de entrada padrão
if st.button('Obter Recomendações'):
    recommendations = get_recommendations(book_input)
    st.write(recommendations)
