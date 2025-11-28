import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Certifique-se de baixar os recursos do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Exemplo de textos brutos
with open(r'C:\Users\gabri\OneDrive\DesenvolvimentodeSoftware2\BIC\ARQUIVOS\Textos_bic.txt', 'r', encoding='utf-8') as f:
    textos_brutos = f.read().lower()  # deixa tudo minúsculo
    textos_brutos = re.sub(r'[^\w\s]', '', textos_brutos)  # remove pontuação

# Separa textos por parágrafo ou outro delimitador
lista_textos = textos_brutos.split("\n\n")

# Stopwords simples (pode expandir)
stopwords = {
    'the','a','an','and','or','but','if','then','else','when','while','of','in','on','at','to',
    'for','from','by','with','about','as','into','like','through','after','over','between',
    'out','against','during','without','before','under','around','among',
    'is','are','was','were','be','been','being',
    'this','that','these','those','it','its','their','my','your','his','her','our',
    'they','them','we','you','he','she','i',
    'do','does','did','doing','can','could','should','would','may','might','must',
    'very','more','most','much','some','any','many','few','several','each','other',
    'one','two','three','all','also','such','so','than','too','not', 'but'
}

lemmatizer = WordNetLemmatizer()
lista_textos_filtrados = []

for texto in lista_textos:
    tokens = word_tokenize(texto, language='english')
    pos_tags = nltk.pos_tag(tokens)

    palavras_filtradas = []
    for word, tag in pos_tags:
        # Converte POS para formato WordNet
        pos = tag[0].upper()
        pos = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(pos, wordnet.NOUN)
        lemma = lemmatizer.lemmatize(word.lower(), pos)

        # Filtra palavras pequenas, não alfabéticas e stopwords
        if len(lemma) > 2 and lemma.isalpha() and lemma not in stopwords:
            palavras_filtradas.append(lemma)

    # Junta as palavras de volta em string
    lista_textos_filtrados.append(" ".join(palavras_filtradas))

# Agora lista_textos_filtrados contém os textos lematizados e limpos
print(lista_textos_filtrados)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Supondo que lista_textos_filtrados já contenha os textos lematizados e filtrados

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(lista_textos_filtrados)

# Cosine similarity
similaridade = cosine_similarity(tfidf_matrix)
print("Cosine similarity:\n", similaridade)

# Jaccard binário
binario = (tfidf_matrix > 0).astype(int)
n = binario.shape[0]
jaccard_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        intersecao = np.minimum(binario[i].toarray(), binario[j].toarray()).sum()
        uniao = np.maximum(binario[i].toarray(), binario[j].toarray()).sum()
        jaccard_matrix[i, j] = intersecao / uniao

# Visualiza a matriz com pandas
df_jaccard = pd.DataFrame(jaccard_matrix, 
                          index=[f'Texto {i+1}' for i in range(n)],
                          columns=[f'Texto {i+1}' for i in range(n)])
print("\nJaccard binário:\n", df_jaccard)

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
import seaborn as sns

jaccard_distance = 1 - jaccard_matrix  # transforma similaridade em distância
jaccard_distance_condensed = squareform(jaccard_distance, checks=False)

Z = linkage(jaccard_distance_condensed, method='average')
plt.figure(figsize=(12, 6))

# Labels formatados com numeração mais clara
labels = [f'Texto {i+1}' for i in range(jaccard_matrix.shape[0])]

dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90,          # ajuda a evitar sobreposição
    leaf_font_size=10,
    color_threshold=0.5
)

plt.title("Dendrograma Hierárquico - Textos")
plt.ylabel("Distância (1 - Jaccard)")
plt.tight_layout()

plt.savefig("dendrograma_textos.png", dpi=300)  # SALVAR ANTES
plt.show()
plt.close()


# ---------------------------
# 2. DEFINIÇÃO DOS CLUSTERS
# ---------------------------

# Distâncias acima da diagonal
tri_upper = jaccard_distance[np.triu_indices_from(jaccard_distance, k=1)]
t = tri_upper.mean()  # média das distâncias

clusters = fcluster(Z, t=t, criterion="distance")

print("Cluster de cada texto:", clusters)


# ---------------------------
# 3. HEATMAP ORDENADO POR CLUSTER
# ---------------------------

# Ordena pelas classes
ordem = np.argsort(clusters)
df_jaccard_ord = df_jaccard.iloc[ordem, ordem]

plt.figure(figsize=(10, 8))

sns.heatmap(
    df_jaccard_ord,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True
)

plt.title("Heatmap da Similaridade Jaccard (Ordenado por Cluster)")
plt.tight_layout()

plt.savefig("MAPADECALOR.png", dpi=300)   # SALVAR ANTES
plt.show()
plt.close()
