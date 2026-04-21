import numpy as np
import pandas as pd

# porta lógica - lê o arquivo e separa as duas primeiras colunas em X e a última coluna em Y
def carregar_porta_logica(caminho):
    dados = pd.read_csv(caminho, header=None, dtype=float).values
 
    X = dados[:, :-1]
    Y = dados[:, -1:]
 
    # print(f"X: {X[0]} \n Y: {Y[0]}")
    return X, Y

# X_and, Y_and = carregar_porta_logica("conjuntos de dados/portas logicas/problemAND.csv")
# print("AND -> X:", X_and.shape, "Y:", Y_and.shape)

# caracteres de Fausett - lê o arquivo e separa as 63 primeiras colunas em X e as 7 últimas colunas em Y
def carregar_fausett(caminho):
    dados = pd.read_csv(caminho, header=None, dtype=float).values
 
    n_entradas = 63
    X = dados[:, :n_entradas]
    Y = dados[:, n_entradas:]
 
    # print(f"X: {X[0]} \n Y: {Y[0]}")
    return X, Y

# X_fausett, Y_fausett = carregar_fausett("conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv")
# print("Fausett -> X:", X_fausett.shape, "Y:", Y_fausett.shape)
  
def carregar_completo_npy(caminho_x, caminho_y):
    X = np.load(caminho_x, allow_pickle=True).astype(float)
    Y = np.load(caminho_y, allow_pickle=True).astype(float)
 
    # transforma X de (n, 10, 12, 1) para (n, 120)
    # cada linha é uma amostra e cada coluna é um pixel (10 * 12 * 1 = 120 pixels por amostra)
    n = X.shape[0]
    X = X.reshape(n, -1)

    # converte Y de 0/1 para -1.0/1.0
    Y = np.interp(Y, [0, 1], [-1.0, 1.0])

    # print(f"X: {X[0]} \n Y: {Y[0]}")
    return X, Y

# X_Completo, Y_Completo = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/X.npy", "conjuntos de dados/CARACTERES COMPLETO/Y_classe.npy")
# print("Completo -> X:", X_Completo.shape, "Y:", Y_Completo.shape)