from Mlp import Mlp
from LeituraArquivo import carregar_completo_npy, carregar_fausett
import numpy as np
map_letras = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

# caminho_dataset = "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
# caminho_dataset_ruido = "./conjuntos de dados/caracteres-Fausett/caracteres-ruido.csv"

# Carrega entradas e saídas
# X, T = carregar_fausett(caminho_dataset)
X, T = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/X.npy", "conjuntos de dados/CARACTERES COMPLETO/Y_classe.npy")

# Carrega entradas e saídas
# X_ruido, T_ruido = carregar_fausett(caminho_dataset_ruido)

rede = Mlp(
    n_inputs=120,
    n_hidden=10,
    n_outputs=26,
    alpha=0.1
)

# Treinamento
historico = rede.train(
    X,
    T,
    epocas=500,
    erro_minimo=0.05
)

# -------------------------
# Teste de uma amostra
# -------------------------
indice_amostra = 0


x = X[indice_amostra]
t = T[indice_amostra]

y = rede.predict(x)

print("\n==============================")
print("RESULTADO DO TESTE")
print("==============================")

print("\nSaída esperada:")
print(t)

print("\nSaída calculada:")
print(y)

print("\nClasse esperada:")
print(map_letras[np.argmax(t)])

print("\nClasse predita:")
print(map_letras[np.argmax(y)])
