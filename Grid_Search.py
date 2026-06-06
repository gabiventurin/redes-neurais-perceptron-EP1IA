from Mlp import Mlp
from LeituraArquivo import carregar_completo_npy
from itertools import product
import os
import sys
from datetime import datetime
os.makedirs("logs", exist_ok=True)

#carrega os dados
X, T = carregar_completo_npy(
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

#seta o hiperparâmetros para o grid search
param_grid = {
    "n_hidden": [10, 20, 40, 60, 100, 120],
    "alpha": [0.001, 0.01, 0.1],
    "epocas" : [500, 1000, 5000],
    "patience": [50, 100],
}

melhor_erro = float("inf")
melhor_rede = None
melhores_parametros = None

combinacoes = list(product(
    param_grid["n_hidden"],
    param_grid["alpha"],
    param_grid["patience"],
    param_grid["epocas"]
))

print(f"Total de combinações: {len(combinacoes)}")

#treina com os hiperparâmetros
for idx, (n_hidden, alpha, patience, epocas) in enumerate(combinacoes, start=1):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nome_log = (
        f"logs/exec_{idx}_"
        f"h{n_hidden}_"
        f"a{alpha}_"
        f"p{patience}_"
        f"e{epocas}_"
        f"{timestamp}.txt"
    )

    # salva console original
    console_original = sys.stdout

    # abre arquivo de log
    log_file = open(nome_log, "w", encoding="utf-8")

    # tudo que for print vai para o arquivo
    sys.stdout = log_file

    try:

        print("\n" + "=" * 60)
        print(f"Combinação {idx}/{len(combinacoes)}")
        print(
            f"n_hidden={n_hidden} | "
            f"alpha={alpha} | "
            f"patience={patience} | "
            f"epocas={epocas}"
        )

        rede = Mlp(
            n_inputs=120,
            n_hidden=n_hidden,
            n_outputs=26,
            alpha=alpha
        )

        resultado = rede.train(
            X,
            T,
            epocas=epocas,
            erro_minimo=0.005,
            X_val=X_val,
            T_val=T_val,
            patience=patience
        )

        erro_val = resultado[1]
        print(f"Melhor erro de validação: {erro_val:.6f}")

        if erro_val < melhor_erro:
            melhor_erro = erro_val
            melhor_rede = rede

            melhores_parametros = {
                "n_hidden": n_hidden,
                "alpha": alpha,
                "patience": patience,
                "epocas": epocas
            }

            print(">>> NOVO MELHOR MODELO")

    finally:
        sys.stdout = console_original
        log_file.close()

    print(f"Execução {idx} salva em: {nome_log}")

#calcula a acurácia do melhor modelo no conjunto de teste
acertos = 0
for x, t in zip(X_test, T_test):
    y = melhor_rede.predict(x)

    if y.argmax() == t.argmax():
        acertos += 1

acuracia = 100 * acertos / len(X_test)

print("MELHOR CONFIGURAÇÃO")
print(f"Erro de validação: {melhor_erro:.6f}")
print(f"Parâmetros: {melhores_parametros}")

print("RESULTADO NO TESTE")
print(f"Acertos: {acertos}/{len(X_test)}")
print(f"Acurácia: {acuracia:.2f}%")