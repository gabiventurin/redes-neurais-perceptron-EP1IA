import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.metrics import confusion_matrix
from Mlp import Mlp

# 1. Configurações de Caminho
pasta_raiz = r'C:\repositorio_IA\redes-neurais-perceptron-EP1IA'
pasta_split = f'{pasta_raiz}/conjuntos de dados/CARACTERES COMPLETO/split'
arquivo_pesos = f'{pasta_raiz}/resultados/20260607_225100/pesos_finais.txt'

# 2. Carregar arquivos
def carregar_npy(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return np.load(caminho, allow_pickle=True)

print("Carregando datasets...")
X_test = carregar_npy(f'{pasta_split}/X_test.npy')
y_test = carregar_npy(f'{pasta_split}/y_test.npy')
X_train = carregar_npy(f'{pasta_split}/X_train.npy')
y_train = carregar_npy(f'{pasta_split}/y_train.npy')

# 3. Definir Arquitetura
n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1] # Deve ser 26 para caracteres
n_hidden = 20
alpha = 0.1

print(f"Arquitetura: {n_inputs} entradas, {n_hidden} escondidos, {n_outputs} saídas.")

# 4. Instanciar Modelo
model = Mlp(n_inputs, n_hidden, n_outputs, alpha)

# 5. Carregar pesos do arquivo .txt bruto
print("Carregando pesos...")
with open(arquivo_pesos, 'r', encoding='utf-8') as f:
    conteudo = f.read()

# Extrair todos os numeros (ignora tudo qeu nao for numero)
numeros_string = re.findall(r"[-+]?\d*\.\d+|\d+", conteudo)
pesos_limpos = np.array([float(n) for n in numeros_string])

# Separar pesos V e W
tamanho_V = n_hidden * (n_inputs + 1)
tamanho_W = n_outputs * (n_hidden + 1)

if len(pesos_limpos) < (tamanho_V + tamanho_W):
    print(f"ERRO: O arquivo tem {len(pesos_limpos)} números, mas a rede precisa de {tamanho_V + tamanho_W}.")
else:
    model.V = pesos_limpos[:tamanho_V].reshape((n_hidden, n_inputs + 1))
    model.W = pesos_limpos[tamanho_V : tamanho_V + tamanho_W].reshape((n_outputs, n_hidden + 1))
    print("Pesos carregados com sucesso!")

# 6. Gerar previsões
y_pred_classes = []
y_true_classes = []

for i in range(len(X_test)):
    saida = model.predict(X_test[i])
    # np.argmax retorna o índice da saída com maior valor (a classe escolhida)
    y_pred_classes.append(np.argmax(saida))
    y_true_classes.append(np.argmax(y_test[i]))

# 7. Matriz de Confusão
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Matriz de Confusão gerada.")

plt.figure(figsize=(15, 12)) # Tamanho maior para 26x26 classes
sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu')
plt.title('Matriz de Confusão - Modelo Log 36')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()