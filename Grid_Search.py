from Mlp import Mlp
from LeituraArquivo import carregar_completo_npy
from itertools import product

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
    "n_hidden": [10],
    "alpha": [0.001],
    "epocas" : [200, 300],
    "patience": [50],
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

    # pega o erro do modelo com os dados de validação dessa combinação
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