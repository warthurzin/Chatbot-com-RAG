"""
Modulo de recuperacao (retrieval) do RAG.

Carrega o indice FAISS e os metadados dos chunks previamente gerados pelo
script build_index.py, e expoe a funcao recuperar(), utilizada pelo grafo
do LangGraph para buscar os trechos mais relevantes para uma pergunta.

O indice e carregado uma unica vez, na primeira chamada (lazy loading),
e reaproveitado nas chamadas seguintes.
"""

import json

import faiss
from sentence_transformers import SentenceTransformer

from app import config

_modelo_embedding = None
_indice_faiss = None
_chunks = None


def _carregar_recursos():
    """Carrega o modelo de embeddings, o indice FAISS e os metadados dos
    chunks, caso ainda nao tenham sido carregados nesta execucao."""
    global _modelo_embedding, _indice_faiss, _chunks

    if _modelo_embedding is None:
        _modelo_embedding = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    if _indice_faiss is None:
        if not config.FAISS_INDEX_PATH.exists():
            raise RuntimeError(
                "Indice FAISS nao encontrado em "
                f"{config.FAISS_INDEX_PATH}. Execute "
                "'python -m app.build_index' antes de iniciar a API."
            )
        _indice_faiss = faiss.read_index(str(config.FAISS_INDEX_PATH))

    if _chunks is None:
        with open(config.CHUNKS_METADATA_PATH, encoding="utf-8") as arquivo:
            _chunks = json.load(arquivo)


def recuperar(pergunta, top_k=None):
    """
    Recupera os chunks mais relevantes para a pergunta informada.

    Retorna uma lista de dicionarios, cada um contendo os metadados do
    chunk (chunk_id, documento_id, titulo, categoria, texto) e o score de
    similaridade (produto interno de vetores normalizados, equivalente a
    similaridade de cosseno).
    """
    _carregar_recursos()

    if top_k is None:
        top_k = config.TOP_K

    vetor_pergunta = _modelo_embedding.encode(
        [pergunta],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = _indice_faiss.search(vetor_pergunta, top_k)

    resultados = []

    for score, indice in zip(scores[0], indices[0]):
        if indice < 0:
            # FAISS retorna -1 quando nao ha vizinhos suficientes.
            continue

        chunk = _chunks[indice].copy()
        chunk["score"] = float(score)
        resultados.append(chunk)

    return resultados