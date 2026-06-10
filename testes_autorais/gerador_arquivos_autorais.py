import numpy as np
import os
from PIL import Image

def processar_imagens_autorais(pasta_origem):
    arquivos = sorted([f for f in os.listdir(pasta_origem) if f.endswith('.png')], 
                      key=lambda x: int(os.path.splitext(x)[0]))
    
    dados = []
    
    print(f"Processando {len(arquivos)} imagens...")
    
    for arq in arquivos:
        caminho = os.path.join(pasta_origem, arq)
        # Abre a imagem e converte para tons de cinza (L)
        img = Image.open(caminho).convert('L')
        
        # Redimensiona para 20x20 (mesmo padrão do seu treino)
        img = img.resize((12, 10)) 
        
        # Converte para array
        img_array = np.array(img)
        
        vetor = (img_array.flatten() < 127).astype(float) 

        # 3. Transforma 0 em -1 e 1 em 1 (Bipolar)
        vetor = np.where(vetor == 0, 1, -1)
        
        dados.append(vetor)
        
    return np.array(dados)

# Caminhos
base_dir = r'C:\repositorio_IA\redes-neurais-perceptron-EP1IA'
pasta_autoral = os.path.join(base_dir, 'conjuntos de dados', 'CARACTERES COMPLETO', 'XAutorais_png')

# Gerar o X autoral
X_autoral = processar_imagens_autorais(pasta_autoral)
np.save(os.path.join(base_dir, 'X_autoral.npy'), X_autoral)

print(f"[OK] X_autoral.npy gerado com sucesso!")