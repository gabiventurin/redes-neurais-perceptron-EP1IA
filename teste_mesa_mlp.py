from Mlp import Mlp
import numpy as np

# INICIALIZA REDE CONFORME O PDF 
# "Apoio a testes de mesa (MLP) Arquivo"

rede = Mlp(
    n_inputs=2,
    n_hidden=3,
    n_outputs=2,
    alpha=0.5
)

# PESOS DEFINIDOS MANUALMENTE CONFORME O PDF 
# "Apoio a testes de mesa (MLP) Arquivo"

rede.V = np.array([
    [-0.1,  0.1, -0.1],
    [-0.1,  0.1,  0.1],
    [ 0.1, -0.1, -0.1]
])

rede.W = np.array([
    [-0.1,  0.1,  0.0,  0.1],
    [ 0.1, -0.1,  0.1, -0.1]
])

# ENTRADA E SAÍDA ESPERADA CONFORME O PDF 
# "Apoio a testes de mesa (MLP) Arquivo"

x = np.array([1, 1])

t = np.array([
    1,
    -1
])

# FORWARD

x_bias, z_in, z_bias, y_in, y = rede.forward(x)

print("\nFORWARD PROPAGATION")

print("\nEntrada:")
print(x)

print("\nSaída esperada:")
print(t)

print("\nz_in:")
print(z_in)

print("\nz:")
print(z_bias[1:])

print("\ny_in:")
print(y_in)

print("\ny:")
print(y)

# SALVA PESOS ANTES DO BP

V_antes = rede.V.copy()
W_antes = rede.W.copy()

print("\nPESOS ANTES DO BACKPROPAGATION")

print("\nMatriz V:")
print(V_antes)

print("\nMatriz W:")
print(W_antes)

# BACKPROPAGATION

rede.backpropagation(
    x_bias,
    z_in,
    z_bias,
    y_in,
    y,
    t
)

# PESOS APÓS O BP

print("\nPESOS APÓS O BACKPROPAGATION")

print("\nMatriz V:")
print(rede.V)

print("\nMatriz W:")
print(rede.W)

# FORWARD ESPERADO DO PDF

print("\nFORWARD ESPERADO (PDF)")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_bias_pdf = np.insert(x, 0, 1)

z_in_pdf = np.dot(V_antes, x_bias_pdf)

z_pdf = sigmoid(z_in_pdf)

z_bias_pdf = np.insert(z_pdf, 0, 1)

y_in_pdf = np.dot(W_antes, z_bias_pdf)

y_pdf = sigmoid(y_in_pdf)

print("\nz esperado:")
print(np.round(z_pdf, 4))

print("\ny esperado:")
print(np.round(y_pdf, 4))