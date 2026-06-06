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


# Carrega entradas e saídas de treino
X, T = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_train.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_train.npy")

# Carrega entrada e saida de validação
X_val, T_val = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_val.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_val.npy")

# Carrega entradas e saídas de teste
X_test, T_test = carregar_completo_npy("conjuntos de dados/CARACTERES COMPLETO/split/X_test.npy", "conjuntos de dados/CARACTERES COMPLETO/split/y_test.npy") 


rede = Mlp(
    n_inputs=120,
    n_hidden=120,
    n_outputs=26,
    alpha=0.01
)

# Treinamento
historico, menor_erro_val = rede.train(
    X,
    T,
    epocas=5000,
    erro_minimo=0.005,
    X_val=X_val,
    T_val=T_val,
    patience=100
)

# -------------------------
# Teste com todo o conjunto de teste
# -------------------------
acertos = 0
total = len(X_test)

for i in range(len(X_test)):
    x = X_test[i]
    t = T_test[i]
    
    y = rede.predict(x)
    
    if np.argmax(t) == np.argmax(y):
        acertos += 1

acuracia = (acertos / total) * 100

print("\n==============================")
print("RESULTADO DO TESTE")
print("==============================")

print(f"\nAcertos: {acertos}/{total}")
print(f"Acurácia: {acuracia:.2f}%")
