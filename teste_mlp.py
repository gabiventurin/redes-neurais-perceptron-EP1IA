from Mlp import Mlp
from LeituraArquivo import carregar_fausett
import numpy as np

caminho_dataset = "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
caminho_dataset_ruido = "./conjuntos de dados/caracteres-Fausett/caracteres-ruido.csv"

# Carrega entradas e saídas
X, T = carregar_fausett(caminho_dataset)

# Carrega entradas e saídas
X_ruido, T_ruido = carregar_fausett(caminho_dataset_ruido)

rede = Mlp(
    n_inputs=63,
    n_hidden=10,
    n_outputs=7,
    alpha=0.1
)

# Treinamento
historico = rede.train(
    X,
    T,
    epocas=5000,
    erro_minimo=0.005
)

# -------------------------
# Teste de uma amostra
# -------------------------
indice_amostra = 1


x = X_ruido[indice_amostra]
t = T_ruido[indice_amostra]

y = rede.predict(x)

print("\n==============================")
print("RESULTADO DO TESTE")
print("==============================")

print("\nSaída esperada:")
print(t)

print("\nSaída calculada:")
print(y)

print("\nClasse esperada:")
print(np.argmax(t))

print("\nClasse predita:")
print(np.argmax(y))
