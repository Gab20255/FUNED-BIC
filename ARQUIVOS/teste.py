from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Seus textos
texto1 = """
the expansion of renewable energy system have accelerate significantly over the past decade , drive by global commitment to reduce carbon emission and transition toward sustainable development . among the various available technology , solar photovoltaics have demonstrate the fast cost decline , largely due to improvement in semiconductor manufacturing and large-scale deployment in emerge economy . current research focus on increase cell efficiency through perovskite-based layer , tandem configuration , and advance light-trapping mechanism . additionally , energy storage solution such a lithium-ion battery and green hydrogen be increasingly integrate into solar farm to stabilize the intermittent production profile typical of photovoltaic system . policymakers be also adopt regulatory framework that encourage grid modernization and the development of distributed generation system in residential and industrial environment . these combine effort aim to enhance energy security , reduce dependency on fossil fuel , and promote new economic opportunity in renewable technology sector
"""

texto2 = """
wind energy continue to play a central role in low-carbon strategy worldwide , largely due to it high scalability and decline operational cost . onshore installation remain the most common , but offshore wind farm have gain attention for their strong and more consistent wind profile . recent study examine the optimization of blade aerodynamics , turbine spacing , and real-time control system to increase efficiency and minimize turbulence effect . another relevant research direction involve hybrid renewable complex combine wind , solar , and storage solution in a single operational unit . this approach allow energy producer to balance generation throughout the day while reduce strain on electrical grid . furthermore , environmental assessment focus on minimize the ecological impact of large-scale installation , particularly regard bird migration route and marine ecosystem . a global energy demand continue to rise , wind power be expect to remain a cornerstone of sustainable energy development .
"""

# Cria o vetor TF-IDF para os textos juntos
vectorizer = TfidfVectorizer( stop_words='english')
tfidf_matrix = vectorizer.fit_transform([texto1, texto2])
vectorizer = TfidfVectorizer(stop_words='english')  # sem limitar palavras
tfidf_matrix = vectorizer.fit_transform([texto1, texto2])  # ambos juntos

similaridade = cosine_similarity(tfidf_matrix)
print(similaridade)
binario = (tfidf_matrix > 0).astype(int)

# Calcula Jaccard
intersecao = np.minimum(binario[0].toarray(), binario[1].toarray()).sum()
uniao = np.maximum(binario[0].toarray(), binario[1].toarray()).sum()
jaccard_tfidf = intersecao / uniao
print("Jaccard com TF-IDF binário:", jaccard_tfidf)
