import pandas as pd
import re
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib

# 0. Carregar modelo spaCy
nlp = spacy.load("pt_core_news_md")

def vetor_spacy(produto_nome):
    doc = nlp(produto_nome.lower())
    return doc.vector  # vetor de 300 dimensões

# 1. Carregar os dados
df = pd.read_csv("/home/denis/.Documents/Desktop/auto_split/src/model/data/training_set_compras.csv")

# 2. Separar features e labels
X_raw = df["IDX"].astype(str)
Y = df[["Duarte", "Dinis", "Carolina"]]

# 3. Vetorização com TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode")
X_tfidf = vectorizer.fit_transform(X_raw)

# 4. Vetores semânticos com spaCy
X_spacy = np.vstack([vetor_spacy(p) for p in X_raw])

# 5. Combinar vetores
from scipy.sparse import hstack
X_combined = hstack([X_tfidf, X_spacy])

# 6. Dividir em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

# 7. Treinar modelo multi-label
model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, Y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# 8. Avaliar
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred, target_names=Y.columns))

# Previsão com fallback semântico
def prever_donos(produto_nome, threshold=0.75, debug=False):

    # Name Normalization
    produto_nome = produto_nome.lower()
    produto_nome = re.sub(r"[^a-zà-ÿ0-9\s]", "", produto_nome).strip()

    tfidf_vetor = vectorizer.transform([produto_nome])
    spacy_vetor = vetor_spacy(produto_nome).reshape(1, -1)

    # Combina os dois vetores
    from scipy.sparse import csr_matrix
    spacy_sparse = csr_matrix(spacy_vetor)
    produto_vetorizado = hstack([tfidf_vetor, spacy_sparse])

    probas = model.predict_proba(produto_vetorizado)

    pred = []
    for i, p in enumerate(probas):
        prob = p[0][1] if len(p[0]) == 2 else p[0][0]
        if debug:
            print(f"[DEBUG] {Y.columns[i]}: prob={prob:.2f}")
        pred.append(int(prob > threshold))

    if sum(pred) == 0:
        max_index = max(range(len(probas)), key=lambda i: probas[i][0][1] if len(probas[i][0]) == 2 else probas[i][0][0])
        pred[max_index] = 1

    return [nome for i, nome in enumerate(Y.columns) if pred[i] == 1]

# 🧪 Exemplos
print(prever_donos("cebola"))
print(prever_donos("tomate"))
print(prever_donos("arroz"))

print(prever_donos("pano de limpeza"))
print(prever_donos("farinha"))
print(prever_donos("açúcar"))

print(prever_donos("blédina"))
print(prever_donos("puré de fruta"))
print(prever_donos("maquiagem"))