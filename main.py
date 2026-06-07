from Mlp import Mlp
from LeituraArquivo import carregar_completo_npy, carregar_fausett
import numpy as np
import os
from datetime import datetime
from gerador_arquivos import (
    salvar_hiperparametros, 
    salvar_pesos_iniciais, 
    salvar_pesos_finais, 
    salvar_historico_erro, 
    salvar_saidas_teste
)
 
# Configuração de pastas de saída
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pasta_saida = f"resultados/{timestamp}"
os.makedirs(pasta_saida, exist_ok=True)

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


# Carrega entradas e saídas de treino
X, T = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_train.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_train.npy")

# Carrega entrada e saida de validação
X_val, T_val = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_val.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_val.npy")

# Carrega entradas e saídas de teste
X_test, T_test = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_test.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_test.npy") 

# Hiperparâmetros
N_INPUTS  = 120
N_HIDDEN  = 20
N_OUTPUTS = 26
ALPHA     = 0.1
EPOCAS    = 1000
ERRO_MIN  = 0.005
PATIENCE  = 100

np.random.seed(42)

# Criação da rede
rede = Mlp(
    n_inputs=N_INPUTS,
    n_hidden=N_HIDDEN,
    n_outputs=N_OUTPUTS,
    alpha=ALPHA
)

# ARQUIVO 1 — Hiperparâmetros da arquitetura
salvar_hiperparametros(
    pasta_saida, N_INPUTS, N_HIDDEN, N_OUTPUTS, ALPHA, 
    EPOCAS, ERRO_MIN, PATIENCE, rede.V.shape, rede.W.shape
)

# ARQUIVO 2 — Pesos iniciais (antes do treino)
V_inicial = rede.V.copy()
W_inicial = rede.W.copy()
salvar_pesos_iniciais(pasta_saida, V_inicial, W_inicial, map_letras)

# Treinamento
historico, menor_erro_val = rede.train(
    X,
    T,
    epocas=EPOCAS,
    erro_minimo=ERRO_MIN,
    X_val=X_val,
    T_val=T_val,
    patience=PATIENCE
)

# ARQUIVO 3 — Pesos finais (após o treino)
salvar_pesos_finais(pasta_saida, rede.V, rede.W, historico, menor_erro_val, map_letras)

# ARQUIVO 4 — Histórico de erro por época
salvar_historico_erro(pasta_saida, historico)

# ARQUIVO 5 — Saídas produzidas no conjunto de teste
acertos, total, acuracia = salvar_saidas_teste(
    pasta_saida, rede, X_test, T_test, map_letras, N_OUTPUTS
)

# Resultado final no console
print("\n==============================")
print("RESULTADO DO TESTE")
print("==============================")
print(f"\nAcertos: {acertos}/{total}")
print(f"Acurácia: {acuracia:.2f}%")
print(f"\nArquivos gerados em: {pasta_saida}/")
print("  1. hiperparametros.txt")
print("  2. pesos_iniciais.txt")
print("  3. pesos_finais.txt")
print("  4. historico_erro.csv")
print("  5. saidas_teste.csv")