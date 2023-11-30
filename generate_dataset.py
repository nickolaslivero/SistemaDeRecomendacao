import pandas as pd
import random


file_path = 'dataset.csv'
dataset_filmes = pd.read_csv(file_path)

usuarios = [
    "Carlos Silva",
    "Ana Souza",
    "Rafael Oliveira",
    "Mariana Santos",
    "Pedro Rodrigues",
    "Juliana Ferreira",
    "Lucas Almeida",
    "Amanda Costa",
    "Fernando Pereira",
    "Camila Gomes",
    "Diego Martins",
    "Laura Bezerra",
    "Tiago Ribeiro",
    "Isabela Fernandes",
    "Ricardo Cardoso",
    "Beatriz Carvalho",
    "Gustavo Melo",
    "Patrícia Nunes",
    "Bruno Cunha",
    "Luiza Castro"
]

dados = {}
for usuario in usuarios:
    filmes_avaliados = random.sample(list(dataset_filmes['book title']), k=random.randint(5, len(dataset_filmes)))
    for filme in dataset_filmes['book title']:
        if filme in filmes_avaliados:
            avaliacao = round(random.uniform(1, 5), 1)
            if usuario not in dados:
                dados[usuario] = {}
            dados[usuario][filme] = avaliacao

dataset_avaliacoes = pd.DataFrame(dados).T.fillna(0)

dataset_avaliacoes.to_csv('dataset_avaliacoes.csv')
