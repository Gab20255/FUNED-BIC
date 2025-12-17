# Projeto de Iniciação Científica — Clusterização de Textos

**Fundação Ezequiel Dias (FUNED)**
**Linguagem:** Python
**Área:** Processamento de Linguagem Natural (PLN) e Análise de Similaridade Textual

---

## 1. Introdução

Este projeto de Iniciação Científica teve como objetivo desenvolver um pipeline computacional para **análise de similaridade semântica e clusterização de textos**, utilizando técnicas consolidadas de **Processamento de Linguagem Natural (PLN)** e **Aprendizado de Máquina não supervisionado**.

A clusterização de textos consiste em agrupar documentos de acordo com sua similaridade temática, sem a necessidade de rótulos prévios. Esse tipo de abordagem é amplamente utilizado em contextos como:

* Organização automática de grandes volumes de textos;
* Identificação de documentos semanticamente redundantes;
* Apoio à revisão bibliográfica;
* Detecção de possíveis casos de plágio;
* Redução do tempo de leitura ao destacar textos altamente semelhantes.

A linguagem **Python** foi escolhida por sua ampla adoção científica e por oferecer um ecossistema robusto de bibliotecas voltadas para **PLN**, **análise de dados**, **visualização científica** e **modelagem matemática**, o que a torna particularmente adequada para esse tipo de aplicação.

---

## 2. Visão Geral do Pipeline

O código implementa um fluxo completo de processamento, dividido nas seguintes etapas:

1. Leitura e pré-processamento dos textos;
2. Tokenização, lematização e filtragem linguística;
3. Vetorização dos textos usando TF-IDF;
4. Cálculo de similaridade (Cosseno e Jaccard);
5. Clusterização hierárquica;
6. Visualização dos resultados por dendrograma e mapa de calor (heatmap).

Cada uma dessas etapas é detalhada a seguir.

---

## 3. Leitura e Pré-processamento dos Textos

Os textos são carregados a partir de um arquivo `.txt`, contendo múltiplos documentos separados por parágrafos. Inicialmente, é realizado um pré-processamento básico:

* Conversão de todo o texto para letras minúsculas;
* Remoção de pontuação e caracteres especiais usando expressões regulares (`re`).

Esse processo reduz ruídos e padroniza os dados, facilitando as etapas posteriores de análise linguística.

---

## 4. Processamento de Linguagem Natural (PLN)

Nesta etapa, são utilizadas funcionalidades da biblioteca **NLTK (Natural Language Toolkit)**, uma das principais bibliotecas de PLN em Python.

### 4.1 Tokenização

Os textos são divididos em palavras (tokens) por meio do método `word_tokenize`, permitindo o tratamento individual de cada termo.

### 4.2 Etiquetagem Morfossintática (POS Tagging)

Cada token recebe uma etiqueta gramatical (substantivo, verbo, adjetivo, advérbio, etc.) com o `nltk.pos_tag`. Essa informação é fundamental para a lematização correta.

### 4.3 Lematização

A lematização é realizada com o `WordNetLemmatizer`, que reduz palavras às suas formas canônicas (lemas), levando em conta sua classe gramatical. Por exemplo:

* *running* → *run*
* *studies* → *study*

Isso contribui para uma representação semântica mais consistente dos textos.

### 4.4 Remoção de Stopwords

Stopwords são palavras muito frequentes que carregam pouco significado semântico (ex.: *the*, *and*, *is*). Elas são removidas utilizando um conjunto definido manualmente, o que permite maior controle sobre o vocabulário analisado.

Além disso, são descartadas palavras:

* Com tamanho menor ou igual a 2 caracteres;
* Que não sejam alfabéticas.

O resultado dessa etapa é uma lista de textos **limpos, lematizados e semanticamente mais informativos**.

---

## 5. Vetorização com TF-IDF

Para permitir cálculos matemáticos de similaridade, os textos são transformados em vetores numéricos usando o método **TF-IDF (Term Frequency–Inverse Document Frequency)**, implementado pelo `TfidfVectorizer` da biblioteca **scikit-learn**.

O TF-IDF pondera a importância de cada termo considerando:

* Sua frequência no documento;
* Sua raridade no conjunto total de textos.

Isso evita que palavras muito comuns dominem a representação vetorial.

---

## 6. Cálculo de Similaridade

### 6.1 Similaridade do Cosseno

A similaridade do cosseno mede o ângulo entre os vetores TF-IDF dos textos. Valores próximos de 1 indicam textos semanticamente semelhantes, enquanto valores próximos de 0 indicam baixa similaridade.

Essa métrica é amplamente utilizada em recuperação de informação e mineração de textos.

### 6.2 Similaridade de Jaccard (Binária)

Também é calculada a similaridade de Jaccard, considerando apenas a presença ou ausência de termos (representação binária).

A similaridade de Jaccard é definida como:

> interseção / união

Ela é particularmente útil para análise estrutural de vocabulário compartilhado entre documentos.

---

## 7. Clusterização Hierárquica

A partir da matriz de similaridade de Jaccard, é construída uma **matriz de distância** (`1 - Jaccard`). Essa matriz é utilizada como entrada para a **clusterização hierárquica aglomerativa**, usando o método `average linkage`, disponível na biblioteca **SciPy**.

Esse tipo de clusterização não exige a definição prévia do número de clusters e permite observar a estrutura hierárquica dos dados.

---

## 8. Visualização dos Resultados

### 8.1 Dendrograma

O dendrograma representa visualmente o processo de agrupamento hierárquico dos textos, permitindo identificar:

* Textos muito semelhantes;
* Grupos temáticos;
* Distâncias entre clusters.

O gráfico é salvo em alta resolução para posterior análise.

### 8.2 Heatmap da Similaridade Jaccard

Por fim, é gerado um **mapa de calor (heatmap)** da matriz de similaridade de Jaccard, ordenado de acordo com os clusters identificados.

Essa visualização facilita a identificação de padrões de similaridade e reforça a validação visual dos agrupamentos encontrados.

---

## 9. Bibliotecas Utilizadas e Justificativa

* **re**: limpeza textual com expressões regulares;
* **NLTK**: tokenização, POS tagging e lematização;
* **WordNet**: base lexical para lematização semântica;
* **NumPy**: manipulação de matrizes numéricas;
* **Pandas**: organização e visualização tabular dos dados;
* **Scikit-learn**: TF-IDF e métricas de similaridade;
* **SciPy**: clusterização hierárquica e cálculo de distâncias;
* **Matplotlib / Seaborn**: visualização científica (dendrogramas e heatmaps).

---

## 10. Considerações Finais

O projeto demonstra como técnicas de PLN aliadas a métodos estatísticos e de aprendizado não supervisionado podem ser utilizadas para extrair conhecimento relevante de grandes volumes de texto. A abordagem adotada é flexível, interpretável e facilmente extensível para outros contextos, como análise de artigos científicos, relatórios técnicos ou bases documentais e



