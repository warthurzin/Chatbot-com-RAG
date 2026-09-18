"""
Schemas Pydantic utilizados pela API para validar requisicoes e formatar
respostas.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Corpo da requisicao enviada pelo front-end ao endpoint /chat."""

    mensagem: str = Field(
        ...,
        min_length=1,
        description="Pergunta ou mensagem do usuario.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Identificador da sessao de conversa, gerado pelo front-end. "
            "Usado para manter a memoria de conversa entre mensagens."
        ),
    )


class FonteResposta(BaseModel):
    """Representa um chunk utilizado como evidencia na resposta."""

    titulo: str
    categoria: str
    score: float


class ChatResponse(BaseModel):
    """Corpo da resposta retornada pelo endpoint /chat."""

    resposta: str
    session_id: str
    score_maximo: float
    fontes: List[FonteResposta] = []


class HealthResponse(BaseModel):
    """Corpo da resposta do endpoint de verificacao de saude da API."""

    status: str
    indice_carregado: bool
    total_chunks: Optional[int] = None