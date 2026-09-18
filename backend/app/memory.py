"""
Memoria de conversa do chatbot.

Mantem, em memoria de processo (um dicionario Python), o historico de
mensagens de cada sessao de conversa, identificada por um session_id
gerado pelo front-end.

Limitacao conhecida: como o historico fica apenas em memoria, ele e
perdido caso o container do backend seja reiniciado. Isso e aceitavel
para o escopo desta atividade e esta documentado no README.
"""

from app import config

# Estrutura: { session_id: [ {"papel": "usuario"|"assistente", "texto": str}, ... ] }
_historicos = {}


def obter_historico(session_id):
    """Retorna a lista de turnos (usuario/assistente) de uma sessao."""
    return _historicos.get(session_id, [])


def adicionar_turno(session_id, papel, texto):
    """
    Adiciona uma mensagem ao historico da sessao e aplica o limite de
    turnos configurado, para que o prompt enviado ao LLM nao cresca
    indefinidamente ao longo de uma conversa longa.
    """
    historico = _historicos.setdefault(session_id, [])
    historico.append({"papel": papel, "texto": texto})

    limite_mensagens = config.MAX_TURNOS_HISTORICO * 2  # usuario + assistente
    if len(historico) > limite_mensagens:
        _historicos[session_id] = historico[-limite_mensagens:]


def formatar_historico(session_id):
    """
    Formata o historico da sessao como texto simples, para ser incluido
    no prompt de geracao da resposta. Retorna string vazia se nao houver
    historico anterior.
    """
    historico = obter_historico(session_id)

    if not historico:
        return ""

    linhas = []
    for turno in historico:
        rotulo = "Usuario" if turno["papel"] == "usuario" else "Assistente"
        linhas.append(f"{rotulo}: {turno['texto']}")

    return "\n".join(linhas)


def limpar_sessao(session_id):
    """Remove o historico de uma sessao especifica."""
    _historicos.pop(session_id, None)