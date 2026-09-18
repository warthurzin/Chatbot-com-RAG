/*
 * Logica do front-end do chatbot.
 *
 * Responsabilidades:
 * - Definir a URL base da API (configuravel via variavel global abaixo).
 * - Gerar/persistir um session_id no navegador (localStorage), para que
 *   o backend consiga manter a memoria de conversa entre mensagens.
 * - Enviar mensagens ao endpoint POST /chat e renderizar a resposta.
 * - Permitir iniciar uma nova conversa (novo session_id + limpeza da
 *   memoria no backend).
 */

// URL base da API. Em desenvolvimento local fora do Docker, ajuste para o
// endereco onde o backend (uvicorn) esta rodando. Quando servido junto
// com o backend via docker-compose, isso pode ser sobrescrito.
const API_BASE_URL = window.CHATBOT_API_BASE_URL || "http://localhost:8000";

const CHAVE_SESSION_ID = "chatbot_filmes_session_id";

const janelaChat = document.getElementById("janela-chat");
const formulario = document.getElementById("formulario-chat");
const campoMensagem = document.getElementById("campo-mensagem");
const botaoEnviar = formulario.querySelector(".botao-enviar");
const botaoNovaConversa = document.getElementById("botao-nova-conversa");
const rodapeStatus = document.getElementById("rodape-status");

function obterOuCriarSessionId() {
  let sessionId = localStorage.getItem(CHAVE_SESSION_ID);

  if (!sessionId) {
    sessionId = gerarIdAleatorio();
    localStorage.setItem(CHAVE_SESSION_ID, sessionId);
  }

  return sessionId;
}

function gerarIdAleatorio() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }

  return `sessao-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function adicionarMensagem(texto, tipo, fontes) {
  const bolha = document.createElement("div");
  bolha.className = `mensagem mensagem-${tipo}`;

  const paragrafo = document.createElement("p");
  paragrafo.textContent = texto;
  bolha.appendChild(paragrafo);

  if (fontes && fontes.length > 0) {
    const listaFontes = document.createElement("div");
    listaFontes.className = "mensagem-fontes";

    const titulosUnicos = [...new Set(fontes.map((f) => f.titulo))];

    titulosUnicos.forEach((titulo) => {
      const marcador = document.createElement("span");
      marcador.textContent = titulo;
      listaFontes.appendChild(marcador);
    });

    bolha.appendChild(listaFontes);
  }

  janelaChat.appendChild(bolha);
  janelaChat.scrollTop = janelaChat.scrollHeight;
}

function definirCarregando(carregando) {
  campoMensagem.disabled = carregando;
  botaoEnviar.disabled = carregando;
  rodapeStatus.textContent = carregando ? "Consultando a base de filmes..." : "";
}

async function enviarMensagem(mensagem) {
  const sessionId = obterOuCriarSessionId();

  const resposta = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mensagem, session_id: sessionId }),
  });

  if (!resposta.ok) {
    const detalhe = await resposta.json().catch(() => null);
    const mensagemErro =
      (detalhe && detalhe.detail) || `Erro ao consultar a API (HTTP ${resposta.status}).`;
    throw new Error(mensagemErro);
  }

  return resposta.json();
}

formulario.addEventListener("submit", async (evento) => {
  evento.preventDefault();

  const texto = campoMensagem.value.trim();
  if (!texto) {
    return;
  }

  adicionarMensagem(texto, "usuario");
  campoMensagem.value = "";
  definirCarregando(true);

  try {
    const dados = await enviarMensagem(texto);
    adicionarMensagem(dados.resposta, "bot", dados.fontes);
  } catch (erro) {
    adicionarMensagem(
      `Nao foi possivel obter uma resposta: ${erro.message}`,
      "erro"
    );
  } finally {
    definirCarregando(false);
    campoMensagem.focus();
  }
});

botaoNovaConversa.addEventListener("click", async () => {
  const sessionId = obterOuCriarSessionId();

  try {
    await fetch(`${API_BASE_URL}/chat/${sessionId}`, { method: "DELETE" });
  } catch (erro) {
    // Falha ao limpar a sessao no backend nao deve impedir o usuario de
    // comecar uma nova conversa no front-end.
    console.warn("Falha ao limpar sessao no backend:", erro);
  }

  localStorage.removeItem(CHAVE_SESSION_ID);
  janelaChat.innerHTML = "";
  adicionarMensagem(
    "Nova conversa iniciada. Pergunte sobre algum filme, diretor ou premio.",
    "sistema"
  );
  campoMensagem.focus();
});

campoMensagem.focus();