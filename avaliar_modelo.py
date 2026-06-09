import os
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from Mlp import Mlp


# Caminhos
pasta_raiz = r"C:\repositorio_IA\redes-neurais-perceptron-EP1IA"
pasta_split = f"{pasta_raiz}/conjuntos de dados/CARACTERES COMPLETO/split"
arquivo_pesos = f"{pasta_raiz}/resultados/20260609_082139/pesos_finais.txt"


def carregar_npy(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    return np.load(caminho, allow_pickle=True)


print("Carregando datasets...")

X_train = carregar_npy(f"{pasta_split}/X_train.npy")
y_train = carregar_npy(f"{pasta_split}/y_train.npy")

X_test = carregar_npy(f"{pasta_split}/X_test.npy")
y_test = carregar_npy(f"{pasta_split}/y_test.npy")

print("\nFormato dos dados")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# Arquitetura da rede
n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

n_hidden = 20
alpha = 0.1

print(
    f"\nArquitetura: {n_inputs} entradas, "
    f"{n_hidden} escondidos, "
    f"{n_outputs} saídas"
)

model = Mlp(
    n_inputs,
    n_hidden,
    n_outputs,
    alpha
)

# Carregamento dos pesos
print("\nCarregando pesos...")

with open(arquivo_pesos, "r", encoding="utf-8") as f:
    conteudo = f.read()

inicio_matriz = conteudo.find("--- Matriz V")

if inicio_matriz != -1:
    conteudo = conteudo[inicio_matriz:]

numeros_string = re.findall(r"[-+]?\d+\.\d+", conteudo)

pesos_limpos = np.array(
    [float(numero) for numero in numeros_string]
)

print("\nQuantidade de números encontrados:", len(pesos_limpos))
print("\nPrimeiros 20 números encontrados:")
print(pesos_limpos[:20])

tamanho_V = n_hidden * (n_inputs + 1)
tamanho_W = n_outputs * (n_hidden + 1)

if len(pesos_limpos) < (tamanho_V + tamanho_W):

    print(
        f"ERRO: O arquivo possui {len(pesos_limpos)} números."
    )

    print(
        f"A rede precisa de {tamanho_V + tamanho_W}."
    )

else:

    model.V = pesos_limpos[:tamanho_V].reshape(
        (n_hidden, n_inputs + 1)
    )

    model.W = pesos_limpos[
        tamanho_V:tamanho_V + tamanho_W
    ].reshape(
        (n_outputs, n_hidden + 1)
    )

    print("Pesos carregados com sucesso!")

# Informações dos pesos
print("\nRESUMO DOS PESOS")

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

print("\nQuantidade de pesos V:", model.V.size)
print("Quantidade de pesos W:", model.W.size)

# Teste
y_pred_classes = []
y_true_classes = []

acertos = 0

for i in range(len(X_test)):

    saida = model.predict(X_test[i])

    classe_predita = np.argmax(saida)
    classe_real = np.argmax(y_test[i])

    y_pred_classes.append(classe_predita)
    y_true_classes.append(classe_real)

    if classe_predita == classe_real:
        acertos += 1

    if i < 10:

        print("\n--------------------")
        print(f"Exemplo {i}")
        print("Real:", classe_real)
        print("Predito:", classe_predita)
        print("Saída:")
        print(np.round(saida, 3))

acuracia = acertos / len(X_test)

print("\nRESULTADO DO TESTE")
print(f"\nAcurácia: {acuracia:.4f}")

# Matriz de confusão
cm = confusion_matrix(
    y_true_classes,
    y_pred_classes
)

print("\nMatriz de Confusão gerada.")

plt.figure(figsize=(15, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    square=True,
    linewidths=0.5
)

plt.title(
    f"Matriz de Confusão - Acurácia {acuracia:.2%}"
)

plt.xlabel("Predito")
plt.ylabel("Real")

plt.tight_layout()
plt.show()