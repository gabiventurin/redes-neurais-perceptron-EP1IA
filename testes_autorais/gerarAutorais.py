import numpy as np

# ==========================================
# CONFIGURAÇÕES DE CAMINHO
# ==========================================

pasta_raiz = r'C:\repositorio_IA\redes-neurais-perceptron-EP1IA'

pasta_caracteres = (
    f'{pasta_raiz}/conjuntos de dados/CARACTERES COMPLETO'
)

# ==========================================
# CARREGAR DADOS AUTORAIS
# ==========================================

print("Carregando arquivos...")

X = np.load(
    f'{pasta_caracteres}/X_autoral.npy',
    allow_pickle=True
)

Y = np.load(
    f'{pasta_caracteres}/Y_autoral.npy',
    allow_pickle=True
)

print("X:", X.shape)
print("Y:", Y.shape)

# ==========================================
# DIMENSÕES DOS CARACTERES
# ==========================================

ALTURA = 12
LARGURA = 10

# ==========================================
# FUNÇÕES DE DESLOCAMENTO
# ==========================================

def deslocar_esquerda(img):

    nova = np.full_like(
        img,
        -1
    )

    nova[:, :-1] = img[:, 1:]

    return nova


def deslocar_direita(img):

    nova = np.full_like(
        img,
        -1
    )

    nova[:, 1:] = img[:, :-1]

    return nova


def deslocar_cima(img):

    nova = np.full_like(
        img,
        -1
    )

    nova[:-1, :] = img[1:, :]

    return nova


def deslocar_baixo(img):

    nova = np.full_like(
        img,
        -1
    )

    nova[1:, :] = img[:-1, :]

    return nova

# ==========================================
# GERAR NOVAS AMOSTRAS
# ==========================================

print("\nGerando variações...")

novos_X = []

novos_Y = []

for i in range(len(X)):

    imagem = X[i].reshape(
        ALTURA,
        LARGURA
    )

    variacoes = [

        imagem,

        deslocar_esquerda(
            imagem
        ),

        deslocar_direita(
            imagem
        ),

        deslocar_cima(
            imagem
        ),

        deslocar_baixo(
            imagem
        )
    ]

    for v in variacoes:

        novos_X.append(
            v.flatten()
        )

        novos_Y.append(
            Y[i]
        )

# ==========================================
# CONVERTER PARA ARRAY
# ==========================================

novos_X = np.array(
    novos_X
)

novos_Y = np.array(
    novos_Y
)

# ==========================================
# SALVAR ARQUIVOS
# ==========================================

caminho_X = (
    f'{pasta_caracteres}/X_autoral_aug.npy'
)

caminho_Y = (
    f'{pasta_caracteres}/Y_autoral_aug.npy'
)

np.save(
    caminho_X,
    novos_X
)

np.save(
    caminho_Y,
    novos_Y
)

# ==========================================
# RESULTADO
# ==========================================

print("\nArquivos gerados com sucesso!")

print(
    "\nNovo X:",
    novos_X.shape
)

print(
    "Novo Y:",
    novos_Y.shape
)

print(
    "\nSalvo em:"
)

print(
    caminho_X
)

print(
    caminho_Y
)

print(
    f"\nTotal de amostras: {len(novos_X)}"
)