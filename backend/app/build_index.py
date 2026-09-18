"""
Script de indexacao: le a base de conhecimento (filmes.json), quebra os
documentos em chunks, gera embeddings e monta um indice vetorial FAISS.

O indice e os metadados dos chunks sao persistidos em disco (pasta
INDEX_DIR), para que a API (retriever.py) apenas carregue o resultado
pronto, sem precisar reprocessar a base a cada inicializacao.

Uso:
    python -m app.build_index
"""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


def carregar_documentos():
    """Le o arquivo JSON com a base de conhecimento de filmes."""
    with open(config.DOCUMENTOS_PATH, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def criar_chunks(documentos):
    """
    Cria um chunk por secao tematica de cada documento (sinopse, ficha
    tecnica, elenco, avaliacao, etc.), em vez de cortar o texto por
    contagem fixa de palavras.

    Motivacao: com uma base de 100 filmes cuja estrutura textual e
    repetitiva (todo documento tem as mesmas secoes: "Ficha Tecnica:",
    "Elenco principal:", "Avaliacao:", etc.), cortar por contagem de
    palavras podia (a) partir uma secao no meio de uma frase e (b)
    misturar o final de uma secao com o inicio de outra dentro do mesmo
    chunk, o que dilui o embedding e faz chunks de filmes diferentes
    ficarem quase indistinguiveis por similaridade vetorial. Manter uma
    secao inteira por chunk evita os dois problemas: cada chunk cobre um
    unico topico e nunca corta uma frase ao meio.

    O titulo do filme e adicionado como prefixo a CADA chunk (nao
    apenas ao primeiro), funcionando como uma "ancora" semantica forte
    que ajuda o modelo de embeddings a associar aquele chunk ao filme
    especifico, mesmo quando o restante do texto e generico (por
    exemplo, "Genero(s): Drama" sozinho poderia remeter a dezenas de
    filmes da base).
    """
    chunks = []

    for doc in documentos:
        prefixo_titulo = f"{doc['titulo']}. "
        secoes = doc.get("secoes", {})

        for numero_chunk, (nome_secao, texto_secao) in enumerate(secoes.items(), start=1):
            if not texto_secao:
                continue

            texto_chunk = f"{prefixo_titulo}{texto_secao}"

            chunks.append(
                {
                    "chunk_id": f'{doc["id"]}_{nome_secao}',
                    "documento_id": doc["id"],
                    "titulo": doc["titulo"],
                    "categoria": doc.get("categoria", ""),
                    "secao": nome_secao,
                    "texto": texto_chunk,
                }
            )

    return chunks


def gerar_embeddings(modelo, chunks):
    """Gera embeddings normalizados (para uso com similaridade de cosseno)."""
    textos = [chunk["texto"] for chunk in chunks]

    embeddings = modelo.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype("float32")


def construir_indice_faiss(embeddings):
    """Cria um indice FAISS de produto interno (equivalente a cosseno,
    pois os vetores estao normalizados)."""
    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatIP(dimensao)
    indice.add(embeddings)
    return indice


def salvar_indice(indice, chunks):
    """Persiste o indice FAISS e os metadados dos chunks em disco."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(indice, str(config.FAISS_INDEX_PATH))

    with open(config.CHUNKS_METADATA_PATH, "w", encoding="utf-8") as arquivo:
        json.dump(chunks, arquivo, ensure_ascii=False, indent=2)


def main():
    print("Carregando documentos de:", config.DOCUMENTOS_PATH)
    documentos = carregar_documentos()
    print(f"Documentos carregados: {len(documentos)}")

    # Verificacao de sanidade: confirma que um filme conhecido esta
    # presente na base, e falha de forma explicita e visivel no log do
    # build (em vez de silenciosamente gerar um indice incompleto ou
    # desatualizado) caso a base tenha vindo vazia ou corrompida.
    titulos = [doc["titulo"] for doc in documentos]
    print(f"Exemplos de titulos carregados: {titulos[:3]}")
    if len(documentos) == 0:
        raise RuntimeError(
            "Nenhum documento foi carregado de "
            f"{config.DOCUMENTOS_PATH}. Verifique se o arquivo "
            "filmes.json existe e nao esta vazio antes de construir o "
            "indice."
        )

    print("Gerando chunks (um por secao tematica de cada filme)...")
    chunks = criar_chunks(documentos)
    print(f"Chunks gerados: {len(chunks)}")

    print("Carregando modelo de embeddings:", config.EMBEDDING_MODEL_NAME)
    modelo = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Gerando embeddings...")
    embeddings = gerar_embeddings(modelo, chunks)
    print("Formato da matriz de embeddings:", embeddings.shape)

    print("Construindo indice FAISS...")
    indice = construir_indice_faiss(embeddings)
    print("Vetores indexados:", indice.ntotal)

    print("Salvando indice e metadados em:", config.INDEX_DIR)
    salvar_indice(indice, chunks)

    print(
        f"Indexacao concluida com sucesso: {len(documentos)} filmes, "
        f"{len(chunks)} chunks, {indice.ntotal} vetores."
    )


if __name__ == "__main__":
    main()