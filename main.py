from HandlerPerceptron import HandlerPerceptron
import numpy as np

def main():

    caminho_entrada = "/home/amandaventurin/Downloads/redes-neurais-perceptron-EP1IA/conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
    caminho_saida = "/home/amandaventurin/Downloads/redes-neurais-perceptron-EP1IA/conjuntos de dados/caracteres-Fausett/saidasFausett.csv"

    indice_perceptron = 0

    # ------------------------------
    # Leitura dos dados
    # ------------------------------
    matriz_entradas = np.loadtxt(
        caminho_entrada,
        delimiter=",",
        encoding="utf-8-sig"
    )

    matriz_saidas = np.loadtxt(
        caminho_saida,
        encoding="utf-8-sig"
    )

    # Seleciona coluna correta da saída
    if matriz_saidas.ndim > 1:
        vetor_saidas = matriz_saidas[:, indice_perceptron]
    else:
        vetor_saidas = matriz_saidas

    # ------------------------------
    # Inicializa perceptron
    # ------------------------------
    num_features = matriz_entradas.shape[1]
    perceptron = HandlerPerceptron(num_features)

    # ------------------------------
    # Primeira amostra
    # ------------------------------
    x = matriz_entradas[0]
    d = vetor_saidas[0]

    # ------------------------------
    # Forward
    # ------------------------------
    y = perceptron.calcular_saida(x)

    # ------------------------------
    # DEBUG (ANTES)
    # ------------------------------
    print("\n--- DEBUG (ANTES) ---")
    print("Entrada:", x)
    print("Pesos:", perceptron.w)
    print("Bias:", perceptron.b)
    print("Taxa de aprendizado:", perceptron.eta)
    print("Saída calculada (y):", y)
    print("Saída esperada (d):", d)

    # ------------------------------
    # Atualização
    # ------------------------------
    if y != d:
        perceptron.atualizar_pesos(x, d)

        # Recalcula saída após ajuste (so para testar o funcionamento)
        y = perceptron.calcular_saida(x)

    # ------------------------------
    # DEBUG (DEPOIS)
    # ------------------------------
    print("\n--- DEBUG (DEPOIS) ---")
    print("Pesos:", perceptron.w)
    print("Bias:", perceptron.b)
    print("Nova saída (y):", y)

    # ------------------------------
    # Resultado final
    # ------------------------------
    if y == d:
        print("\nResultado: Classificação CORRETA")
    else:
        print("\nResultado: Classificação INCORRETA")

if __name__ == "__main__":
    main()

