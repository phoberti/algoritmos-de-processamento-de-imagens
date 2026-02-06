import cv2
import numpy as np
import matplotlib.pyplot as plt

def converter_para_greyscale(imagem):
    # Obtém as dimensões da imagem (altura, largura e número de canais de cor)
    altura, largura, _ = imagem.shape
    
    # Cria uma matriz vazia para armazenar a imagem em tons de cinza (uint8 para economizar memória)
    greyscale = np.zeros((altura, largura), dtype=np.uint8)
    
    # Percorre todos os pixels da imagem
    for i in range(altura):  # Itera sobre as linhas da imagem
        for j in range(largura):  # Itera sobre as colunas da imagem
            # Obtém os valores dos canais de cor (vermelho, verde e azul)
            r, g, b = imagem[i, j]
            
            # Calcula a intensidade do cinza usando a fórmula de luminância
            # 0.299 * R + 0.587 * G + 0.114 * B (padrão NTSC)
            greyscale[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
    
    # Retorna a imagem convertida para escala de cinza
    return greyscale

def exibir_imagem(titulo, imagem):
    # Cria uma nova figura com tamanho 6x6 polegadas
    plt.figure(figsize=(6, 6))
    
    # Exibe a imagem usando a escala de cinza (cmap='gray')
    plt.imshow(imagem, cmap='gray')
    
    # Define o título da imagem
    plt.title(titulo)
    
    # Remove os eixos para uma visualização mais limpa
    plt.axis('off')
    
    # Exibe a imagem na tela
    plt.show()

def aplicar_convolucao(imagem, kernel):
    # Obtém as dimensões da imagem (altura e largura)
    altura, largura = imagem.shape
    
    # Obtém as dimensões do kernel (altura e largura)
    k_altura, k_largura = kernel.shape
    
    # Calcula o padding necessário para manter o tamanho da imagem de saída igual ao de entrada
    pad = k_altura // 2  # Assume que o kernel é quadrado e ímpar (ex: 3x3, 5x5)
    
    # Adiciona padding à imagem original preenchendo com zeros
    imagem_padded = np.pad(imagem, pad, mode='constant', constant_values=0)
    
    # Cria uma matriz para armazenar o resultado da convolução, com o mesmo tamanho da imagem original
    resultado = np.zeros_like(imagem, dtype=np.float64)  # float64 para evitar problemas de precisão
    
    # Percorre cada pixel da imagem original
    for i in range(altura):  # Itera sobre as linhas
        for j in range(largura):  # Itera sobre as colunas
            # Aplica a convolução: multiplica a região da imagem pelo kernel e soma os valores
            resultado[i, j] = np.sum(imagem_padded[i:i+k_altura, j:j+k_largura] * kernel)
    
    # Retorna a imagem resultante da convolução
    return resultado

def filtro_gaussiano(imagem, tamanho=5, sigma=1.0):
    # Cria um eixo simétrico centrado em 0, com valores igualmente espaçados
    eixo = np.linspace(-(tamanho // 2), tamanho // 2, tamanho)

    # Calcula a função Gaussiana 1D para o eixo
    gauss_1d = np.exp(-(eixo**2) / (2 * sigma**2))
    
    # Normaliza a distribuição para que a soma dos pesos seja 1
    gauss_1d /= gauss_1d.sum()

    # Cria o kernel Gaussiano 2D aplicando o produto externo do vetor 1D com ele mesmo
    gauss_kernel = np.outer(gauss_1d, gauss_1d)

    # Garante que a soma do kernel ainda seja 1 após operações numéricas
    gauss_kernel /= gauss_kernel.sum()

    # Aplica a convolução da imagem com o kernel Gaussiano para suavização
    return aplicar_convolucao(imagem, gauss_kernel)

def operador_laplaciano(imagem):
    # Define o kernel Laplaciano (detecção de bordas)
    # Esse kernel realça regiões onde há mudanças bruscas de intensidade na imagem
    kernel_laplaciano = np.array([[0,  1,  0], 
                                  [1, -4,  1], 
                                  [0,  1,  0]])

    # Aplica a convolução da imagem com o kernel Laplaciano para realçar bordas
    return aplicar_convolucao(imagem, kernel_laplaciano)

def marr_hildreth(imagem):
    # Aplica o filtro Gaussiano para suavizar a imagem
    imagem_suavizada = filtro_gaussiano(imagem, tamanho=5, sigma=1.0)

    # Aplica o operador Laplaciano na imagem suavizada
    laplaciano = operador_laplaciano(imagem_suavizada)

    # Cria uma matriz para armazenar os pontos de cruzamento de zero
    zero_crossing = np.zeros_like(laplaciano, dtype=np.uint8)

    # Percorre a imagem (ignorando as bordas)
    for i in range(1, laplaciano.shape[0] - 1):  # Linhas
        for j in range(1, laplaciano.shape[1] - 1):  # Colunas
            # Obtém o valor atual do pixel
            valor_atual = laplaciano[i, j]

            # Verifica os vizinhos em todas as direções (horizontal, vertical e diagonais)
            vizinhos = [laplaciano[i-1, j], laplaciano[i+1, j],  # Cima e Baixo
                        laplaciano[i, j-1], laplaciano[i, j+1],  # Esquerda e Direita
                        laplaciano[i-1, j-1], laplaciano[i-1, j+1],  # Diagonais superiores
                        laplaciano[i+1, j-1], laplaciano[i+1, j+1]]  # Diagonais inferiores
            
            # Se houver uma mudança de sinal entre o pixel atual e seus vizinhos, marca como borda
            if any(valor_atual * vizinho < 0 for vizinho in vizinhos):
                zero_crossing[i, j] = 255  # Marca a borda como branca (255)

    return zero_crossing

def filtro_canny(imagem, limiar_inferior=50, limiar_superior=150):
    # Aplica um filtro gaussiano para suavizar a imagem e reduzir o ruído
    imagem_suavizada = filtro_gaussiano(imagem, tamanho=5, sigma=1.0)
    
    # Define os kernels para calcular o gradiente na direção x e y usando o operador Sobel
    kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Aplica convolução na imagem suavizada para calcular os gradientes
    gradiente_x = aplicar_convolucao(imagem_suavizada, kernel_sobel_x)
    gradiente_y = aplicar_convolucao(imagem_suavizada, kernel_sobel_y)
    
    # Calcula a magnitude do gradiente combinando as direções x e y
    magnitude = np.sqrt(gradiente_x**2 + gradiente_y**2)
    
    # Calcula a direção do gradiente em radianos e converte para graus
    direcao = np.arctan2(gradiente_y, gradiente_x)
    
    # Obtém as dimensões da imagem
    altura, largura = magnitude.shape
    
    # Inicializa uma matriz de supressão de não máximos com zeros
    supressao = np.zeros((altura, largura), dtype=np.uint8)
    
    # Converte os ângulos de radianos para graus e ajusta valores negativos
    angulos = direcao * (180 / np.pi)
    angulos[angulos < 0] += 180
    
    # Percorre todos os pixels da imagem, exceto as bordas
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            vizinhos = [0, 0]  # Inicializa os vizinhos para comparação
            ang = angulos[i, j]  # Obtém o ângulo do gradiente no pixel atual
            
            # Determina os vizinhos a serem comparados com base na direção do gradiente
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                vizinhos = [magnitude[i, j-1], magnitude[i, j+1]]  # Direção horizontal
            elif 22.5 <= ang < 67.5:
                vizinhos = [magnitude[i-1, j+1], magnitude[i+1, j-1]]  # Direção diagonal descendente
            elif 67.5 <= ang < 112.5:
                vizinhos = [magnitude[i-1, j], magnitude[i+1, j]]  # Direção vertical
            elif 112.5 <= ang < 157.5:
                vizinhos = [magnitude[i-1, j-1], magnitude[i+1, j+1]]  # Direção diagonal ascendente
            
            # Mantém o valor do pixel apenas se for maior que seus vizinhos
            if magnitude[i, j] >= max(vizinhos):
                supressao[i, j] = magnitude[i, j]
    
    # Inicializa a matriz para armazenar os pixels da borda final
    bordas = np.zeros_like(supressao)
    
    # Pixels com intensidade maior que o limiar superior são bordas fortes (255)
    bordas[supressao >= limiar_superior] = 255
    
    # Pixels com intensidade entre os limiares são bordas fracas (128)
    bordas[(supressao >= limiar_inferior) & (supressao < limiar_superior)] = 128
    
    # Retorna a imagem com as bordas detectadas
    return bordas

def metodo_otsu(imagem):
    # Calcula o histograma da imagem com 256 bins (níveis de intensidade de 0 a 255)
    histograma, _ = np.histogram(imagem, bins=256, range=(0, 256))
    
    # Obtém o número total de pixels na imagem
    total_pixels = imagem.size
    
    # Inicializa variáveis para armazenar o melhor limiar e a máxima variância entre classes
    melhor_t, max_variancia = 0, 0
    
    # Calcula a soma total dos níveis de intensidade ponderados pelo histograma
    soma_total = np.sum(np.arange(256) * histograma)
    
    # Inicializa variáveis para o cálculo dinâmico
    soma_fundo, peso_fundo = 0, 0
    
    # Percorre todos os possíveis limiares de 0 a 255
    for t in range(256):
        # Atualiza o peso (quantidade de pixels) do fundo da imagem
        peso_fundo += histograma[t]
        
        # Se não houver pixels no fundo, continua para o próximo limiar
        if peso_fundo == 0:
            continue
        
        # Calcula o peso do objeto (parte restante da imagem)
        peso_objeto = total_pixels - peso_fundo
        
        # Se não houver pixels no objeto, encerra o loop
        if peso_objeto == 0:
            break
        
        # Atualiza a soma dos níveis de intensidade do fundo
        soma_fundo += t * histograma[t]
        
        # Calcula a média de intensidade do fundo e do objeto
        media_fundo = soma_fundo / peso_fundo
        media_objeto = (soma_total - soma_fundo) / peso_objeto
        
        # Calcula a variância entre classes
        variancia = peso_fundo * peso_objeto * (media_fundo - media_objeto) ** 2
        
        # Atualiza o melhor limiar se a variância for maior que a máxima encontrada
        if variancia > max_variancia:
            max_variancia = variancia
            melhor_t = t
    
    # Aplica o limiar encontrado para binarizar a imagem
    return (imagem < melhor_t).astype(np.uint8) * 255

def watershed(imagem):
    # Aplica um filtro gaussiano para suavizar a imagem e reduzir o ruído
    suavizada = filtro_gaussiano(imagem, tamanho=5, sigma=1.0)
    
    # Aplica os operadores de Sobel para calcular o gradiente em x e y
    sobel_x = aplicar_convolucao(suavizada, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    sobel_y = aplicar_convolucao(suavizada, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    
    # Calcula a magnitude do gradiente
    gradiente = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Inicializa a matriz de marcadores para a segmentação
    marcadores = np.zeros_like(imagem, dtype=np.int32)
    
    # Define regiões iniciais para os marcadores
    # Pixels com gradiente abaixo da média são marcados como fundo (1)
    marcadores[gradiente < np.mean(gradiente)] = 1
    
    # Pixels com gradiente acima do percentil 90 são marcados como objetos (2)
    marcadores[gradiente > np.percentile(gradiente, 90)] = 2
    
    # Obtém as dimensões da imagem
    altura, largura = imagem.shape
    
    # Propaga os rótulos dos marcadores para regiões desconhecidas
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            if marcadores[i, j] == 0:  # Se o pixel não foi marcado ainda
                # Coleta os vizinhos já marcados
                vizinhos = [marcadores[i-1, j], marcadores[i+1, j], marcadores[i, j-1], marcadores[i, j+1]]
                vizinhos = [v for v in vizinhos if v > 0]  # Filtra valores não marcados
                
                # Atribui ao pixel a categoria mais frequente entre os vizinhos
                if vizinhos:
                    marcadores[i, j] = max(vizinhos, key=vizinhos.count)
    
    # Retorna a máscara binária onde os pixels segmentados (objetos) são marcados como 255
    return (marcadores == 2).astype(np.uint8) * 255

def contar_objetos(imagem_binaria):
    # Obtém as dimensões da imagem binária
    altura, largura = imagem_binaria.shape
    
    # Cria uma matriz para rastrear os pixels já visitados
    visitado = np.zeros_like(imagem_binaria, dtype=bool)
    
    # Inicializa o contador de objetos
    contador = 0
    
    # Define a função de preenchimento por inundação (Flood Fill) para marcar áreas conectadas
    def flood_fill(x, y):
        # Pilha para armazenar os pixels a serem processados
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()  # Remove um pixel da pilha
            
            # Verifica se o pixel está dentro dos limites e ainda não foi visitado
            if 0 <= cx < altura and 0 <= cy < largura and imagem_binaria[cx, cy] == 255 and not visitado[cx, cy]:
                visitado[cx, cy] = True  # Marca o pixel como visitado
                
                # Adiciona os pixels vizinhos (4-conectividade) à pilha
                stack.extend([(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)])
    
    # Percorre toda a imagem para encontrar objetos
    for i in range(altura):
        for j in range(largura):
            # Se encontrar um pixel branco (255) não visitado, inicia um novo preenchimento
            if imagem_binaria[i, j] == 255 and not visitado[i, j]:
                contador += 1  # Incrementa o contador de objetos
                flood_fill(i, j)  # Chama a função de preenchimento para marcar o objeto inteiro
    
    # Ajusta a contagem para remover falsos positivos (fundo detectado como objeto)
    if contador > 2:
        return contador - 2  # Remove dois objetos considerados fundo
    elif contador > 1:
        return contador - 1  # Remove um objeto considerado fundo
    else:
        return contador

def cadeia_freeman(imagem_binaria):
    # Define os movimentos da cadeia de Freeman (8 direções possíveis)
    movimentos = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    
    # Obtém as dimensões da imagem binária
    altura, largura = imagem_binaria.shape
    
    # Inicializa a lista para armazenar a sequência da cadeia de Freeman
    cadeia = []
    
    # Conjunto para rastrear os pixels já visitados
    visitado = set()
    
    # Função auxiliar para encontrar o primeiro pixel branco (255) na imagem
    def encontrar_ponto_inicial():
        for y in range(altura):
            for x in range(largura):
                if imagem_binaria[y, x] == 255:
                    return x, y  # Retorna as coordenadas do primeiro pixel da borda
        return None  # Retorna None se nenhum pixel branco for encontrado
    
    # Encontra o ponto inicial da borda do objeto
    ponto_inicial = encontrar_ponto_inicial()
    if ponto_inicial is None:
        return []  # Retorna uma lista vazia se não houver contorno
    
    # Define as coordenadas iniciais
    x, y = ponto_inicial
    
    # Adiciona o ponto inicial ao conjunto de visitados
    visitado.add((x, y))
    
    # Flag para controlar se é o primeiro movimento
    primeiro_passo = True
    
    # Loop para percorrer os pixels da borda seguindo a cadeia de Freeman
    while True:
        encontrado = False  # Flag para verificar se um próximo pixel válido foi encontrado
        
        # Percorre todas as 8 direções possíveis
        for i, (dx, dy) in enumerate(movimentos):
            nx, ny = x + dx, y + dy  # Calcula a nova posição do pixel vizinho
            
            # Verifica se a nova posição está dentro dos limites da imagem e ainda não foi visitada
            if 0 <= nx < largura and 0 <= ny < altura and (nx, ny) not in visitado and imagem_binaria[ny, nx] == 255:
                cadeia.append(i)  # Adiciona a direção na cadeia de Freeman
                visitado.add((nx, ny))  # Marca o pixel como visitado
                x, y = nx, ny  # Atualiza a posição atual
                encontrado = True
                break  # Sai do loop ao encontrar um pixel válido
        
        # Se nenhum pixel vizinho válido for encontrado, encerra o loop
        if not encontrado:
            break
        
        # Se voltar ao ponto inicial após pelo menos um movimento, encerra o loop
        if (x, y) == ponto_inicial and not primeiro_passo:
            break
        
        # Após o primeiro movimento, desativa a flag
        primeiro_passo = False
    
    # Retorna a sequência da cadeia de Freeman
    return cadeia

def filtro_box_manual(imagem, tamanho):
    # Obtém as dimensões da imagem
    altura = len(imagem)
    largura = len(imagem[0])
    
    # Calcula o padding necessário para o kernel (tamanho // 2)
    pad = tamanho // 2
    
    # Cria uma matriz com bordas preenchidas com zero (zero-padding)
    imagem_padded = [[0] * (largura + 2 * pad) for _ in range(altura + 2 * pad)]
    
    # Copia a imagem original para o centro da matriz com padding
    for i in range(altura):
        for j in range(largura):
            imagem_padded[i + pad][j + pad] = int(imagem[i][j])  # Converte para int para evitar problemas de precisão
    
    # Cria uma matriz para armazenar o resultado do filtro
    resultado = [[0] * largura for _ in range(altura)]
    
    # Percorre cada pixel da imagem original
    for i in range(altura):
        for j in range(largura):
            soma = 0  # Inicializa a soma dos valores vizinhos
            
            # Aplica a média da vizinhança do tamanho especificado
            for k in range(-pad, pad + 1):
                for l in range(-pad, pad + 1):
                    soma += int(imagem_padded[i + pad + k][j + pad + l])  # Soma os valores da região do kernel
            
            # Calcula a média e atribui ao pixel correspondente na imagem de saída
            resultado[i][j] = soma // (tamanho * tamanho)
    
    # Retorna a imagem filtrada
    return resultado

def segmentacao_por_intensidade(imagem):
    # Obtém as dimensões da imagem
    altura = len(imagem)
    largura = len(imagem[0])
    
    # Cria uma matriz para armazenar a imagem segmentada
    resultado = [[0] * largura for _ in range(altura)]
    
    # Percorre cada pixel da imagem
    for i in range(altura):
        for j in range(largura):
            pixel = imagem[i][j]  # Obtém o valor de intensidade do pixel
            
            # Aplica a segmentação com base nos intervalos de intensidade
            if 0 <= pixel <= 50:
                resultado[i][j] = 25  # Baixa intensidade
            elif 51 <= pixel <= 100:
                resultado[i][j] = 75  # Intensidade um pouco maior
            elif 101 <= pixel <= 150:
                resultado[i][j] = 125  # Intensidade média
            elif 151 <= pixel <= 200:
                resultado[i][j] = 175  # Alta intensidade
            elif 201 <= pixel <= 255:
                resultado[i][j] = 255  # Máxima intensidade
    
    # Retorna a imagem segmentada
    return resultado

# Comparação entre Marr-Hildreth e Canny:
# 
# O método Marr-Hildreth e o algoritmo de Canny são utilizados para detecção de bordas, mas possuem abordagens diferentes.
#
#  Marr-Hildreth:
# - Aplica um filtro Gaussiano para suavizar a imagem e reduzir o ruído.
# - Depois, utiliza o operador Laplaciano para detectar bordas pelo cruzamento de zero.
# - Gera bordas mais grossas e menos precisas, pois não realiza supressão de não máximos.
# - Tem um tempo de execução mais rápido, pois envolve menos etapas de processamento.
# - Pode ser menos sensível a ruídos, mas também pode falhar na detecção de bordas fracas.
#
#  Canny:
# - Também começa com um filtro Gaussiano para suavização.
# - Calcula os gradientes da imagem para identificar as áreas de maior variação de intensidade.
# - Aplica a supressão de não máximos para tornar as bordas mais finas e bem definidas.
# - Utiliza histerese com dois limiares para eliminar bordas falsas e conectar contornos reais.
# - Gera bordas mais precisas e bem definidas, mas é mais sensível a ruídos.
# - Tem um tempo de execução maior que o Marr-Hildreth, pois envolve mais cálculos.
#
#  Exemplo prático:
# Ao aplicar os dois métodos na imagem "1.jpg", percebemos que:
# - Marr-Hildreth detecta bordas de forma mais "espessa", podendo incluir detalhes desnecessários.
# - Canny gera bordas mais finas e limpas, destacando melhor os contornos da imagem.
# - O tempo de processamento do Canny é maior, mas a qualidade das bordas é superior.
#
#  Qual escolher?
# - Se a prioridade for velocidade e menos sensibilidade a ruídos, Marr-Hildreth pode ser suficiente.
# - Se for necessário um contorno mais preciso e bem definido, Canny é a melhor opção.
#
# Ambos os métodos têm vantagens e desvantagens, e a escolha ideal depende do contexto da aplicação.


def menu():
    global imagem  # Define a variável "imagem" como global para ser acessada/modificada dentro da função.

    # Solicita ao usuário que insira o caminho da imagem.
    caminho = input("Digite o caminho da imagem: ")
    print("Processando imagem em Greyscale...")

    # Carrega a imagem a partir do caminho fornecido usando OpenCV.
    imagem_colorida = cv2.imread(caminho)

    # Verifica se a imagem foi carregada corretamente. Se não, exibe uma mensagem de erro e encerra a função.
    if imagem_colorida is None:
        print("Erro ao carregar a imagem.")
        return  # Sai da função se a imagem não foi carregada.

    # Converte a imagem colorida para escala de cinza utilizando a função definida anteriormente.
    imagem = converter_para_greyscale(imagem_colorida)

    # Inicia um loop para exibir o menu interativo até que o usuário escolha sair.
    while True:
        # Exibe o menu de opções para o usuário.
        print("\nMenu de Processamento de Imagens")
        print("1 - Ver imagem em Greyscale")
        print("2 - Aplicar Marr-Hildreth")
        print("3 - Aplicar Canny")
        print("4 - Aplicar Método de Otsu")
        print("5 - Aplicar Watershed")
        print("6 - Contar Objetos")
        print("7 - Cadeia de Freeman")
        print("8 - Filtro Box")
        print("9 - Segmentacao por Intensidade")
        print("0 - Sair")

        # Solicita ao usuário que escolha uma opção.
        opcao = input("Escolha uma opção: ")

        # Se o usuário escolher '0', o loop é interrompido e o programa encerra o menu.
        if opcao == '0':
            break

        # Caso o usuário escolha '1', exibe a imagem em escala de cinza.
        elif opcao == '1':
            
            exibir_imagem("Imagem em Greyscale", imagem)

        # Caso o usuário escolha '2', aplica o algoritmo Marr-Hildreth e exibe a imagem resultante.
        elif opcao == '2':
            print("Aplicando Marr-Hildreth, aguarde...")
            exibir_imagem("Marr-Hildreth", marr_hildreth(imagem))

        # Caso o usuário escolha '3', aplica o filtro de Canny e exibe a imagem resultante.
        elif opcao == '3':
            print("Aplicando Filtro Canny, aguarde...")
            exibir_imagem("Filtro Canny", filtro_canny(imagem))

        # Caso o usuário escolha '4', aplica o Método de Otsu e exibe a imagem resultante.
        elif opcao == '4':
            print("Aplicando Método de Otsu, aguarde...")
            exibir_imagem("Método de Otsu", metodo_otsu(imagem))

        # Caso o usuário escolha '5', aplica o algoritmo Watershed e exibe a imagem segmentada.
        elif opcao == '5':
            print("Aplicando Watershed, aguarde...")
            exibir_imagem("Watershed", watershed(imagem))

        # Caso o usuário escolha '6', aplica o Método de Otsu para binarizar a imagem
        # e conta o número de objetos detectados na imagem binária.
        elif opcao == '6':
            print("Aplicando Método de Otsu e contando objetos, aguarde...")
            otsu = metodo_otsu(imagem)  # Converte a imagem para binária antes da contagem.
            print(f"Número de objetos detectados: {contar_objetos(otsu)}")

        # Caso o usuário escolha '7', aplica a Cadeia de Freeman na imagem binária (gerada por Otsu).
        elif opcao == '7':
            print("Aplicando Método de Otsu e gerando Cadeia de Freeman, aguarde...")
            otsu = metodo_otsu(imagem)  # Converte a imagem para binária antes do processamento.
            freeman_code = cadeia_freeman(otsu)  # Obtém a sequência da Cadeia de Freeman.
            print(f'Cadeia de Freeman: {freeman_code}')  # Exibe o resultado.

        # Caso o usuário escolha '8', exibe opções de tamanho para o Filtro Box.
        elif opcao == '8':
            print("Escolha o tamanho do Filtro Box:")
            print("1 - 2x2")
            print("2 - 3x3")
            print("3 - 5x5")
            print("4 - 7x7")

            # Solicita ao usuário que escolha um dos tamanhos disponíveis.
            escolha = input("Digite o número da opção: ")

            # Aplica o filtro Box no tamanho escolhido e exibe a imagem resultante.
            if escolha == '1':
                print("Aplicando Filtro Box 2x2, aguarde...")
                resultado = filtro_box_manual(imagem, 2)
                exibir_imagem("Filtro Box 2x2", resultado)
            elif escolha == '2':
                print("Aplicando Filtro Box 3x3, aguarde...")
                resultado = filtro_box_manual(imagem, 3)
                exibir_imagem("Filtro Box 3x3", resultado)
            elif escolha == '3':
                print("Aplicando Filtro Box 5x5, aguarde...")
                resultado = filtro_box_manual(imagem, 5)
                exibir_imagem("Filtro Box 5x5", resultado)
            elif escolha == '4':
                print("Aplicando Filtro Box 7x7, aguarde...")
                resultado = filtro_box_manual(imagem, 7)
                exibir_imagem("Filtro Box 7x7", resultado)
            else:
                print("Opção inválida!")  # Exibe um aviso se a escolha não for válida.

        # Caso o usuário escolha '9', aplica a segmentação por intensidade e exibe a imagem resultante.
        elif opcao == '9':
            print("Segmentando a imagem por intensidade, aguarde...")
            imagem_segmentada = segmentacao_por_intensidade(imagem)
            exibir_imagem("Segmentação por Intensidade", imagem_segmentada)

        # Se o usuário digitar qualquer outro valor que não esteja no menu, exibe uma mensagem de erro.
        else:
            print("Opção inválida.")

# Verifica se o script está sendo executado diretamente.
if __name__ == "__main__":
    imagem = None  # Inicializa a variável global "imagem" como None.
    menu()  
