import numpy as np
import os

# Número de classes
n_classes = 26

# Cria uma matriz de identidade 26x26
# Cada linha é um vetor one-hot: [1,0,0...], [0,1,0...], etc.
y_autoral = np.eye(n_classes)

# Salva o arquivo no formato que o numpy usa
np.save('Y_autoral.npy', y_autoral)

print("[OK] Y_autoral.npy gerado! Agora o modelo tem como comparar a saída.")