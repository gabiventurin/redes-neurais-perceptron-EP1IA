from Mlp import Mlp
from LeituraArquivo import carregar_completo_npy

from itertools import product

import pandas as pd
import numpy as np

import os
import time

# PASTA DE SAÍDA
PASTA_SAIDA = "analise_hiperparametros"

os.makedirs(PASTA_SAIDA, exist_ok=True)

# CARREGA DADOS
X_train, T_train = carregar_completo_npy(
    "conjuntos de dados/CARACTERES COMPLETO/split/X_train.npy",
    "conjuntos de dados/CARACTERES COMPLETO/split/y_train.npy"
)

X_val, T_val = carregar_completo_npy(
    "conjuntos de dados/CARACTERES COMPLETO/split/X_val.npy",
    "conjuntos de dados/CARACTERES COMPLETO/split/y_val.npy"
)

X_test, T_test = carregar_completo_npy(
    "conjuntos de dados/CARACTERES COMPLETO/split/X_test.npy",
    "conjuntos de dados/CARACTERES COMPLETO/split/y_test.npy"
)

 
# HIPERPARÂMETROS (setar os parametros que nao vao variar com os melhores valores #definidos pelo grid:
# alpha: 0,1 ; epocas: 5000 ; patience: 100; n_hidden: 20
param_grid = {
    "n_hidden": [20],
    "alpha": [ 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    "epocas": [5000],
    "patience": [100]
}

# HIPERPARÂMETRO AVALIADO (é usado no titulo do arquivo de saida)
hiperp = 'alpha'

combinacoes = list(product(
    param_grid["n_hidden"],
    param_grid["alpha"],
    param_grid["patience"],
    param_grid["epocas"]
))
print(f"\nIniciando execução")
print(f"\nTotal de combinações: {len(combinacoes)}")

# RESULTADOS
resultados = []

# LOOP PRINCIPAL
for idx, (n_hidden, alpha, patience, epocas) in enumerate(combinacoes, start=1):

    print("\n" + "=" * 60)
    print(f"Execução {idx}/{len(combinacoes)}")

    print(
        f"n_hidden={n_hidden} | "
        f"alpha={alpha} | "
        f"patience={patience} | "
        f"epocas={epocas}"
    )

    inicio = time.perf_counter()

    rede = Mlp(
        n_inputs=120,
        n_hidden=n_hidden,
        n_outputs=26,
        alpha=alpha
    )

    historico_erro, menor_erro_val, _ = rede.train(
        X_train,
        T_train,
        epocas=epocas,
        erro_minimo=0.005,
        X_val=X_val,
        T_val=T_val,
        patience=patience
    )

    tempo_execucao = time.perf_counter() - inicio

    # TESTE

    acertos = 0

    for x, t in zip(X_test, T_test):

        y = rede.predict(x)

        if np.argmax(y) == np.argmax(t):
            acertos += 1

    acuracia = (
        acertos / len(X_test)
    ) * 100

    # EARLY STOPPING
    epocas_executadas = len(historico_erro)

    early_stopping = epocas_executadas < epocas

    # SALVA RESULTADO
    resultados.append({

        "n_hidden": n_hidden,

        "alpha": alpha,

        "patience": patience,

        "epocas_maximas": epocas,

        "epocas_executadas": epocas_executadas,

        "early_stopping": early_stopping,

        "tempo_execucao_segundos": tempo_execucao,

        "erro_validacao": menor_erro_val,

        "erro_treinamento_final": historico_erro[-1],

        "acuracia_teste": acuracia

    })


# DATAFRAME
df = pd.DataFrame(resultados)

# ORDENA PELO ERRO DE VALIDAÇÃO
df = df.sort_values(
    by="erro_validacao"
)

# SALVA CSV

arquivo_csv = (
    f"{PASTA_SAIDA}/resultado_variacao_{hiperp}csv"
)

df.to_csv(
    arquivo_csv,
    index=False
)



print("\nCSV salvo em:")
print(arquivo_csv)
