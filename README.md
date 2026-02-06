# Processamento de Imagens Digitais

## Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de **Processamento de Imagens Digitais**, no curso de **Ciência da Computação – UNIOESTE (Universidade Estadual do Oeste do Paraná)**.

O programa permite aplicar diferentes técnicas de processamento e segmentação de imagens por meio de um menu interativo no terminal.

---

## Requisitos do Sistema

É necessário ter o **Python** instalado na máquina.

Bibliotecas utilizadas:

* OpenCV
* NumPy
* Matplotlib

Para instalar as dependências, utilize:

```bash
pip install opencv-python numpy matplotlib
```

---

## Como Executar o Programa

### Windows

No terminal da IDE ou do sistema, execute:

```bash
python main.py
```

### Linux

No terminal da IDE ou do sistema, execute:

```bash
python3 main.py
```

Observação:
Após inserir o caminho da imagem, o programa pode levar até 1 minuto para iniciar o processamento, pois a conversão para grayscale é aplicada automaticamente.

---

## Funcionalidades do Programa

Após carregar a imagem, o sistema exibe um menu interativo com as seguintes opções:

### 1 - Ver imagem em Greyscale

Converte a imagem para escala de cinza.

### 2 - Aplicar Marr-Hildreth

Detecta bordas utilizando o operador Laplaciano do Gaussiano (LoG).

### 3 - Aplicar Canny

Realiza detecção de bordas utilizando o algoritmo de Canny.

### 4 - Aplicar Método de Otsu

Aplica limiarização automática baseada na variância intra-classe.

### 5 - Aplicar Watershed

Executa segmentação baseada no algoritmo Watershed.

### 6 - Contar Objetos

Identifica e contabiliza objetos presentes na imagem.

### 7 - Cadeia de Freeman

Extrai a cadeia de contorno utilizando o método de Freeman.

### 8 - Filtro Box

Aplica suavização utilizando Filtro Box nos seguintes tamanhos:

* 2x2
* 3x3
* 5x5
* 7x7

### 9 - Segmentação por Intensidade

Realiza segmentação com base na intensidade dos pixels.

### 0 - Sair

Encerra o programa.

Observação:
Algumas operações podem levar até 1 minuto para serem concluídas, dependendo do tamanho da imagem e do processamento realizado.

---


## Contexto Acadêmico

Projeto desenvolvido com fins acadêmicos para aplicação prática de técnicas de:

* Detecção de bordas
* Segmentação de imagens
* Filtragem espacial
* Limiarização
* Extração de contornos
