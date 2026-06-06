import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from LeituraArquivo import carregar_fausett
from Mlp import Mlp


def testar_rede(rede, X_teste, T_teste):

    total_acertos = 0

    classes_reais = []
    classes_preditas = []

    for x, t in zip(X_teste, T_teste):

        y = rede.predict(x)

        classe_predita = np.argmax(y)
        classe_real = np.argmax(t)

        classes_preditas.append(classe_predita)
        classes_reais.append(classe_real)

        if classe_predita == classe_real:
            total_acertos += 1

    acuracia = total_acertos / len(X_teste)

    return (
        acuracia,
        classes_reais,
        classes_preditas
    )


def grid_search(
    X_treino,
    T_treino,
    X_teste,
    T_teste,
    lista_alpha,
    lista_erro_minimo,
    n_hidden=10,
    epocas=5000
):

    resultados = []

    melhor_rede = None
    melhor_historico = None

    melhor_acuracia = -1

    melhores_classes_reais = None
    melhores_classes_preditas = None

    for alpha in lista_alpha:

        for erro_minimo in lista_erro_minimo:

            print(
                f"\nTreinando: "
                f"alpha={alpha} "
                f"erro_minimo={erro_minimo}"
            )

            rede = Mlp(
                n_inputs=X_treino.shape[1],
                n_hidden=n_hidden,
                n_outputs=T_treino.shape[1],
                alpha=alpha
            )

            historico = rede.train(
                X_treino,
                T_treino,
                epocas=epocas,
                erro_minimo=erro_minimo
            )

            (
                acuracia,
                classes_reais,
                classes_preditas
            ) = testar_rede(
                rede,
                X_teste,
                T_teste
            )

            resultados.append(
                {
                    "alpha": alpha,
                    "erro_minimo": erro_minimo,
                    "epocas_executadas": len(historico),
                    "erro_final": historico[-1],
                    "acuracia": acuracia * 100
                }
            )

            if acuracia > melhor_acuracia:

                melhor_acuracia = acuracia

                melhor_rede = rede
                melhor_historico = historico

                melhores_classes_reais = classes_reais
                melhores_classes_preditas = classes_preditas

    return (
        pd.DataFrame(resultados),
        melhor_rede,
        melhor_historico,
        melhores_classes_reais,
        melhores_classes_preditas
    )


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================

X_treino, T_treino = carregar_fausett(
    "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
)

X_teste, T_teste = carregar_fausett(
    "./conjuntos de dados/caracteres-Fausett/caracteres-ruido.csv"
)

# ==================================================
# GRID SEARCH
# ==================================================

(
    tabela_resultados,
    melhor_rede,
    melhor_historico,
    classes_reais,
    classes_preditas
) = grid_search(
    X_treino,
    T_treino,
    X_teste,
    T_teste,
    lista_alpha=[0.001, 0.05, 0.1, 0.2],
    lista_erro_minimo=[0.05],
    n_hidden=10,
    epocas=5000
)

# ==================================================
# TABELA DE RESULTADOS
# ==================================================

print("\nRESULTADOS:")
print(tabela_resultados)

# ==================================================
# GRÁFICO 1
# ERRO MÉDIO X ÉPOCA
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(melhor_historico)

plt.title("Erro Médio x Época")
plt.xlabel("Época")
plt.ylabel("Erro Médio")

plt.grid(True)

plt.savefig("erro_vs_epoca.png")
plt.show()

# ==================================================
# GRÁFICO 2
# ALPHA X ACURÁCIA
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(
    tabela_resultados["alpha"],
    tabela_resultados["acuracia"],
    marker="o"
)

plt.title("Alpha x Acurácia")
plt.xlabel("Alpha")
plt.ylabel("Acurácia (%)")

plt.grid(True)

plt.savefig("alpha_vs_acuracia.png")
plt.show()

# ==================================================
# GRÁFICO 3
# MATRIZ DE CONFUSÃO
# ==================================================

matriz = confusion_matrix(
    classes_reais,
    classes_preditas
)

plt.figure(figsize=(8, 8))

plt.imshow(matriz)

plt.colorbar()

plt.title("Matriz de Confusão")

plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

for i in range(matriz.shape[0]):
    for j in range(matriz.shape[1]):
        plt.text(
            j,
            i,
            matriz[i, j],
            ha="center",
            va="center"
        )

plt.savefig("matriz_confusao.png")
plt.show()
