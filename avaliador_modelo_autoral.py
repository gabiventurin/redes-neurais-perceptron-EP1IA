import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from sklearn.metrics import confusion_matrix

from Mlp import Mlp

# 1. Configurações de Caminho

pasta_raiz = r'C:\repositorio_IA\redes-neurais-perceptron-EP1IA'
pasta_caracteres = f'{pasta_raiz}/conjuntos de dados/CARACTERES COMPLETO'
pasta_split = f'{pasta_raiz}/conjuntos de dados/CARACTERES COMPLETO/split'
arquivo_pesos = f'{pasta_raiz}/resultados/20260609_082139/pesos_finais.txt'

# 2. Carregar arquivos

def carregar_npy(caminho):

    if not os.path.exists(caminho):

        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    return np.load(
        caminho,
        allow_pickle=True
    )

print("Carregando datasets...")

X_test = carregar_npy(
    f'{pasta_caracteres}/X_autoral_aug.npy'
)

y_test = carregar_npy(
    f'{pasta_caracteres}/Y_autoral_aug.npy'
)

X_train = carregar_npy(
    f'{pasta_split}/X_train.npy'
)

y_train = carregar_npy(
    f'{pasta_split}/y_train.npy'
)

print("\nFormato dos dados")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# 3. Definir Arquitetura

n_inputs = X_train.shape[1]

n_outputs = y_train.shape[1]

n_hidden = 20

alpha = 0.1

print(
    f"\nArquitetura: "
    f"{n_inputs} entradas, "
    f"{n_hidden} escondidos, "
    f"{n_outputs} saídas"
)

# 4. Instanciar Modelo

model = Mlp(
    n_inputs,
    n_hidden,
    n_outputs,
    alpha
)

# 5. Carregar pesos

print("\nCarregando pesos...")

with open(
    arquivo_pesos,
    'r',
    encoding='utf-8'
) as f:

    conteudo = f.read()

inicio_matriz = conteudo.find(
    "--- Matriz V"
)

if inicio_matriz != -1:

    conteudo = conteudo[
        inicio_matriz:
    ]

# EXTRAI APENAS NÚMEROS DECIMAIS

numeros_string = re.findall(
    r"[-+]?\d+\.\d+",
    conteudo
)

pesos_limpos = np.array(
    [
        float(n)
        for n in numeros_string
    ]
)

print(
    "\nQuantidade de números encontrados:",
    len(pesos_limpos)
)

print(
    "\nPrimeiros 20 números encontrados:"
)

print(
    pesos_limpos[:20]
)
# Tamanho esperado das matrizes

tamanho_V = (
    n_hidden
    * (n_inputs + 1)
)

tamanho_W = (
    n_outputs
    * (n_hidden + 1)
)

if len(pesos_limpos) < (tamanho_V + tamanho_W):

    print(
        f"ERRO: O arquivo possui "
        f"{len(pesos_limpos)} números."
    )

    print(
        f"A rede precisa de "
        f"{tamanho_V + tamanho_W}."
    )

else:

    model.V = pesos_limpos[
        :tamanho_V
    ].reshape(
        (
            n_hidden,
            n_inputs + 1
        )
    )

    model.W = pesos_limpos[
        tamanho_V:
        tamanho_V + tamanho_W
    ].reshape(
        (
            n_outputs,
            n_hidden + 1
        )
    )

    print("Pesos carregados com sucesso!")

# Diagnóstico dos pesos

print("\n")
print("RESUMO DOS PESOS")

print(
    "\nV -> min:",
    np.min(model.V),
    "| max:",
    np.max(model.V)
)

print(
    "W -> min:",
    np.min(model.W),
    "| max:",
    np.max(model.W)
)

print(
    "\nQuantidade de pesos V:",
    model.V.size
)

print(
    "Quantidade de pesos W:",
    model.W.size
)

# 6. Gerar previsões

y_pred_classes = []

y_true_classes = []

acertos = 0

for i in range(len(X_test)):

    saida = model.predict(
        X_test[i]
    )

    classe_predita = np.argmax(
        saida
    )

    classe_real = np.argmax(
        y_test[i]
    )

    y_pred_classes.append(
        classe_predita
    )

    y_true_classes.append(
        classe_real
    )

    if classe_predita == classe_real:

        acertos += 1

    # DEBUG DOS PRIMEIROS EXEMPLOS

    if i < 10:

        print("\n-")

        print(
            f"Exemplo {i}"
        )

        print(
            "Real:",
            classe_real
        )

        print(
            "Predito:",
            classe_predita
        )

        print(
            "Saída:"
        )

        print(
            np.round(
                saida,
                3
            )
        )

# Acurácia

acuracia = (
    acertos
    / len(X_test)
)

print("\n")
print("RESULTADO DO TESTE")

print(
    f"\nAcurácia: {acuracia:.4f}"
)

# 7. Matriz de Confusão

cm = confusion_matrix(
    y_true_classes,
    y_pred_classes
)

print(
    "\nMatriz de Confusão gerada."
)

plt.figure(
    figsize=(15, 12)
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    square=True,
    linewidths=0.5
)

plt.title(
    f'Matriz de Confusão - '
    f'Acurácia {acuracia:.2%}'
)

plt.xlabel(
    'Predito'
)

plt.ylabel(
    'Real'
)

plt.tight_layout()
plt.show()