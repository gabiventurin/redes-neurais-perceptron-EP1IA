from Mlp import Mlp
from LeituraArquivo import carregar_fausett
import numpy as np

caminho_dataset = "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"

# Carrega automaticamente X e T
# X -> entradas
# T -> saídas esperadas
X, T = carregar_fausett(caminho_dataset)

x = X[0]
t = T[0]

rede = Mlp(
    n_inputs=63,
    n_hidden=10,
    n_outputs=7,
    alpha=0.1,
)

y = rede.train(X, T, epocas=5000, erro_minimo = 0.005)

print(y)