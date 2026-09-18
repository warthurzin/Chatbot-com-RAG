"""
Configuracao central da aplicacao.

Todas as variaveis sensiveis ou dependentes de ambiente (chave de API,
nome do modelo, caminhos de arquivos) sao lidas aqui, a partir de variaveis
de ambiente. Em desenvolvimento local, elas podem ser carregadas de um
arquivo .env (nao versionado). Em Docker, sao passadas pelo docker-compose.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variaveis do arquivo .env, se existir (uso em desenvolvimento local).
# Em producao/Docker, as variaveis ja vem definidas no ambiente do container.
load_dotenv()

# Diretorio base do backend (pai da pasta app/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Chave de API da Groq. Obrigatoria para a geracao de respostas.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Modelo da Groq utilizado na etapa de geracao da resposta final.
#
# Observacao: o modelo originalmente especificado para esta atividade era
# qwen/qwen3.6-27b. Durante o desenvolvimento, a Groq descontinuou esse
# modelo (modelos "preview" da Groq podem ser descontinuados a qualquer
# momento, sem aviso previo) e passou a retornar erro 404 (model_not_found)
# para ele. O projeto foi atualizado para usar qwen/qwen3.8-27b, sucessor
# direto do modelo original na mesma familia. Mais detalhes no README.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Modelo de embeddings utilizado para indexar e consultar o FAISS.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Caminhos de dados e indice.
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DOCUMENTOS_PATH = DATA_DIR / "filmes.json"

INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index"))
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_METADATA_PATH = INDEX_DIR / "chunks.json"

# Observacao sobre chunking: o pipeline de indexacao (build_index.py)
# cria um chunk por secao tematica de cada filme (sinopse, ficha
# tecnica, elenco, avaliacao), em vez de cortar o texto por contagem
# fixa de palavras. Por isso nao ha parametros de tamanho/overlap de
# chunk configuraveis aqui; a granularidade e definida pela propria
# estrutura de secoes gerada em backend/data/coletar_tmdb.py.

# Parametros de recuperacao (retrieval).
#
# TOP_K foi aumentado de 4 para 8 apos a expansao da base de 20 para 100
# filmes: com mais documentos candidatos, um valor maior de TOP_K reduz
# a chance de o chunk correto (por exemplo, o que contem a duracao ou o
# elenco de um filme especifico) ficar de fora da janela considerada
# pelo roteamento condicional (decidir_evidencia) e pelo prompt de
# geracao. Em particular, quando um filme tem 9 secoes (sinopse, ano,
# direcao, genero, duracao, orcamento, bilheteria, elenco, avaliacao),
# um TOP_K pequeno pode deixar de fora a secao especifica perguntada
# quando ela compete de perto com outras secoes do mesmo filme.
TOP_K = int(os.getenv("TOP_K", "8"))
LIMIAR_EVIDENCIA = float(os.getenv("LIMIAR_EVIDENCIA", "0.35"))

# Parametros de memoria de conversa.
MAX_TURNOS_HISTORICO = int(os.getenv("MAX_TURNOS_HISTORICO", "6"))

# Parametros da chamada ao LLM.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
#
# LLM_MAX_TOKENS precisa ser generoso: modelos de raciocinio como o
# qwen3 podem consumir uma parte significativa do orcamento de tokens em
# raciocinio interno antes de produzir a resposta visivel, mesmo com
# reasoning_effort="none" (que nao e garantido para todos os modelos e
# versoes na Groq). Um valor baixo aqui pode resultar em respostas
# vazias (finish_reason="length"), especialmente com prompts mais
# longos, como os desta aplicacao apos a expansao da base para 100
# filmes (mais contexto recuperado = prompt mais longo = mais tokens de
# raciocinio consumidos).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))

# CORS: origens permitidas para o front-end acessar a API.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


def validar_configuracao():
    """Valida configuracoes essenciais e lanca erro claro se algo faltar."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY nao foi definida. Configure a variavel de ambiente "
            "GROQ_API_KEY (ver arquivo .env.example) antes de iniciar a "
            "aplicacao."
        )