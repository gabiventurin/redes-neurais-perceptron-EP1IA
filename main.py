from HandlerPerceptron import HandlerPerceptron
import numpy as np

def main():

    caminho_entrada = "./conjuntos de dados/caracteres-Fausett/caracteres-limpo.csv"
    caminho_saida = "./conjuntos de dados/caracteres-Fausett/saidasFausett.csv"

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

    # PRIMEIRO LAÇO: itera sobre os neurônios (1 neurônio por letra)
    for indice_perceptron in range(matriz_saidas.shape[1]):
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

        mudanca_neuronio = False # inicializa variável para o segundo laço

        # SEGUNDO LAÇO: itera sobre as epocas do neuronio
        while not mudanca_neuronio:
            mudanca_neuronio = True;

            #TERCEIRO LAÇO: itera para cada valor da saída (s:t)
            for i in range(vetor_saidas.shape[0]):
                x = matriz_entradas[i]
                d = vetor_saidas[i]

                # ------------------------------
                # Forward
                # ------------------------------
                y = perceptron.calcular_saida(x)

                # ------------------------------
                # DEBUG
                # ------------------------------
                print("\n--- DEBUG ---")
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
                    mudanca_neuronio = False; # assim, será necessário passar por mais uma época
                    perceptron.atualizar_pesos(x, d)
                    print("\nResultado: Classificação INCORRETA")
                else:
                    print("\nResultado: Classificação CORRETA")
    

if __name__ == "__main__":
    main()

