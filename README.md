# Cineasta - Chatbot com RAG sobre Filmes

Chatbot web com Retrieval-Augmented Generation (RAG) sobre uma base de
conhecimento de 100 filmes, orquestrado com LangGraph, busca vetorial
com FAISS e geracao de resposta final com a Groq (modelo
`qwen/qwen3.8-27b`).

## Equipe

- Arthur Marques da Silveira
- Pedro Henrique Leal Amaral
- Luanna Evellyn Batista da Silva
- Marielly de Araujo Silva

## Sumario

- [Dominio e base de conhecimento](#dominio-e-base-de-conhecimento)
- [Arquitetura](#arquitetura)
- [Pipeline de RAG](#pipeline-de-rag)
- [Estrutura do repositorio](#estrutura-do-repositorio)
- [Pre-requisitos](#pre-requisitos)
- [Configuracao das variaveis de ambiente](#configuracao-das-variaveis-de-ambiente)
- [Como executar com Docker (recomendado)](#como-executar-com-docker-recomendado)
- [Como executar localmente sem Docker](#como-executar-localmente-sem-docker)
- [Como executar no GitHub Codespaces](#como-executar-no-github-codespaces)
- [Regerando a base de conhecimento](#regerando-a-base-de-conhecimento)
- [Uso da interface](#uso-da-interface)
- [Decisoes e observacoes tecnicas](#decisoes-e-observacoes-tecnicas)
- [Limitacoes conhecidas](#limitacoes-conhecidas)

## Dominio e base de conhecimento

O dominio escolhido foi **filmes**. A base de conhecimento tem **100
filmes**, coletados de forma programatica a partir da API publica do
**TMDb (The Movie Database)**, garantindo dados reais e verificaveis,
com sinopses em portugues do Brasil.

- Fonte dos dados: [TMDb API v3](https://developer.themoviedb.org/reference/intro/getting-started)
- Script de coleta: [`backend/data/coletar_tmdb.py`](backend/data/coletar_tmdb.py)
- Base gerada (versionada no repositorio): [`backend/data/filmes.json`](backend/data/filmes.json)
- Cada filme na base registra tambem o campo `fonte_url`, com o link
  direto para a pagina do filme no TMDb, permitindo auditar a origem
  de qualquer informacao.

Cada filme e representado por 9 fatos (secoes) independentes: sinopse,
ano de lancamento, direcao, genero, duracao, orcamento, bilheteria,
elenco principal e avaliacao media de usuarios.

Este produto usa a API do TMDb, mas nao e endossado ou certificado
pelo TMDb, conforme exigido pelos
[termos de uso da API](https://www.themoviedb.org/documentation/api/terms-of-use).

## Arquitetura

```
[ Navegador ]
      |
      | HTTP (fetch)
      v
[ Frontend: nginx + HTML/CSS/JS ]  (porta 5500)
      |
      | HTTP (POST /chat)
      v
[ Backend: FastAPI ]  (porta 8000)
      |
      |-- LangGraph (orquestracao do fluxo de RAG)
      |     |-- reescrita de pergunta (Groq, quando ha historico)
      |     |-- recuperacao vetorial (FAISS + sentence-transformers)
      |     |-- roteamento condicional (com/sem evidencia)
      |     |-- montagem de contexto
      |     |-- geracao da resposta final (Groq)
      |
      |-- Memoria de conversa (em memoria de processo, por sessao)
      |
      v
[ API externa: Groq (qwen/qwen3.8-27b) ]
```

Frontend e backend rodam em containers Docker separados, comunicando-se
por HTTP.

## Pipeline de RAG

1. O script `coletar_tmdb.py` busca 100 filmes ja lancados e com
   avaliacao consolidada (minimo de 1000 votos no TMDb), combinando
   ordenacao por popularidade e por nota media, para gerar uma base
   diversa e com dados confiaveis.
2. Para cada filme, cada fato (sinopse, direcao, genero, duracao,
   orcamento, bilheteria, elenco, avaliacao, ano de lancamento) e
   escrito como uma frase de linguagem natural completa e autocontida,
   e nao como um rotulo tecnico solto (por exemplo, "O filme Cidade de
   Deus foi dirigido por Fernando Meirelles." em vez de "Direcao:
   Fernando Meirelles."). O nome do filme e repetido dentro de cada
   frase. Essa decisao de formato foi resultado de testes praticos e
   esta detalhada na secao "Decisoes e observacoes tecnicas".
3. O script `build_index.py` cria um chunk por fato (nao corta o texto
   por contagem de palavras): cada filme gera 9 chunks, um por secao.
4. Cada chunk e transformado em um embedding vetorial usando o modelo
   multilingue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
5. Os embeddings sao armazenados em um indice vetorial FAISS
   (`IndexFlatIP`, equivalente a similaridade de cosseno com vetores
   normalizados).
6. O indice e gerado automaticamente durante o build da imagem Docker
   do backend, a partir do `filmes.json` versionado no repositorio (nao
   e necessario nenhum passo manual de indexacao antes de subir os
   containers).
7. Quando o usuario envia uma pergunta pela interface, o backend recebe
   a mensagem e o identificador da sessao de conversa.
8. Se ja houver historico na sessao, a pergunta e reescrita por uma
   chamada a LLM (Groq) para ficar autossuficiente (por exemplo,
   substituindo "ele" ou "esse filme" pelo nome especifico mencionado
   anteriormente na conversa). Isso melhora a qualidade da busca
   vetorial em perguntas de continuidade.
9. A pergunta (reescrita ou original) e usada para buscar os 8 chunks
   mais relevantes no indice FAISS.
10. Um roteamento condicional no grafo do LangGraph verifica se o score
    de similaridade do melhor resultado ultrapassa um limiar minimo
    (0.35). Se nao ultrapassar, o fluxo desvia para uma resposta padrao
    de abstencao, sem chamar a LLM e sem exibir nenhuma fonte.
11. Se houver evidencia suficiente, os chunks recuperados sao
    combinados em um contexto e enviados, junto com o historico da
    conversa, para a Groq gerar a resposta final.
12. A resposta e devolvida a interface junto com a(s) fonte(s) (titulo
    do filme) efetivamente citada(s) pela LLM no formato [Fonte X]; se
    nenhuma fonte for citada (por exemplo, na resposta de abstencao),
    nenhuma fonte e exibida.

Todo esse fluxo (reescrita, recuperacao, decisao condicional, montagem
de contexto, geracao) e implementado como um grafo do LangGraph em
`backend/app/rag_graph.py`.

## Estrutura do repositorio

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # API FastAPI (endpoints /chat, /health)
│   │   ├── config.py             # Configuracao via variaveis de ambiente
│   │   ├── models.py             # Schemas Pydantic
│   │   ├── rag_graph.py          # Grafo LangGraph (RAG conversacional)
│   │   ├── retriever.py          # Busca vetorial no indice FAISS
│   │   ├── memory.py             # Memoria de conversa por sessao
│   │   └── build_index.py        # Script de indexacao (chunking + FAISS)
│   ├── data/
│   │   ├── filmes.json           # Base de conhecimento (100 filmes, TMDb)
│   │   ├── coletar_tmdb.py       # Script que coleta e gera filmes.json
│   │   └── requirements-dados.txt # Dependencia extra so para coletar_tmdb.py
│   ├── requirements.txt
│   └── Dockerfile
│   └── .dockerignore
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── config.js                 # URL da API (ajustavel por ambiente)
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## Pre-requisitos

- Docker e Docker Compose (plugin `docker compose`), para o metodo de
  execucao recomendado.
- Python 3.12+ e `pip`, apenas se for executar sem Docker.
- Uma chave de API da Groq (gratuita, sem cartao de credito). Crie em
  console.groq.com, na secao "API Keys".
- Uma chave de API do TMDb, apenas se for regerar a base de dados
  (`filmes.json` ja vem pronto no repositorio, entao isso e opcional).
  Crie gratuitamente em themoviedb.org/settings/api.

## Configuracao das variaveis de ambiente

Antes de qualquer metodo de execucao, copie o arquivo de exemplo e
preencha sua chave da Groq:

```bash
cp .env.example .env
```

Edite o `.env` e substitua `coloque_sua_chave_groq_aqui` pela sua chave
real (formato `gsk_...`):

| Variavel        | Obrigatoria | Padrao               | Descricao                                             |
|-----------------|:-----------:|-----------------------|--------------------------------------------------------|
| `GROQ_API_KEY`  | Sim         | -                      | Chave de API da Groq, usada na geracao da resposta.     |
| `GROQ_MODEL`    | Nao         | `qwen/qwen3.8-27b`     | Modelo da Groq utilizado na geracao da resposta final.  |
| `CORS_ORIGINS`  | Nao         | `*`                    | Origens permitidas para chamadas do frontend a API.     |

Outros parametros do pipeline (`TOP_K`, `LIMIAR_EVIDENCIA`,
`MAX_TURNOS_HISTORICO`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) tambem sao
configuraveis por variavel de ambiente; os valores padrao e o
raciocinio por tras de cada um estao documentados diretamente em
`backend/app/config.py`.

## Como executar com Docker (recomendado)

Este e o metodo de execucao principal, testado e recomendado para uso
local (fora do GitHub Codespaces; para Codespaces, ver a secao
especifica mais abaixo).

1. Clone o repositorio e entre na pasta do projeto.

2. Configure o `.env` conforme a secao anterior.

3. Suba os containers:

   ```bash
   docker compose up --build
   ```

   Esse comando builda a imagem do backend (instala dependencias e
   gera o indice FAISS automaticamente a partir de `filmes.json`, ja
   incluido no repositorio) e a imagem do frontend (nginx servindo os
   arquivos estaticos), e sobe os dois servicos.

   Durante o build do backend, o log deve mostrar algo como:

   ```
   Documentos carregados: 100
   Chunks gerados: 900
   Indexacao concluida com sucesso: 100 filmes, 900 chunks, 900 vetores.
   ```

4. Acesse a aplicacao:

   - Frontend (chat): http://localhost:5500
   - Backend (API): http://localhost:8000
   - Verificacao de saude da API: http://localhost:8000/health
   - Documentacao interativa da API: http://localhost:8000/docs

   O arquivo frontend/config.js já aponta, por padrão, para http://localhost:8000, que é o endereço correto quando os containers rodam localmente. Ainda assim, é necessário garantir que a porta 8000 (backend) esteja configurada como pública e não como privada na aba PORTS do ambiente (clique com o botão direito na porta 8000 e selecione Port Visibility > Public), caso contrário o navegador não conseguirá se comunicar com a API mesmo com a URL correta configurada.

5. Para parar os containers:

   ```bash
   docker compose down
   ```

6. Caso precise reconstruir do zero (por exemplo, apos regenerar
   `filmes.json`), garanta que nao ha imagens ou cache antigos
   interferindo:

   ```bash
   docker compose down
   docker builder prune -a -f
   docker compose up --build
   ```

## Como executar localmente sem Docker

Util para desenvolvimento e depuracao do backend isoladamente.

1. Configure o `.env` conforme a secao de variaveis de ambiente (na
   raiz do repositorio; o backend le esse arquivo mesmo rodando fora de
   um container).

2. Crie e ative um ambiente virtual, e instale as dependencias:

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   No Windows, ative o ambiente virtual com `venv\Scripts\activate`.

3. Gere o indice FAISS a partir da base de conhecimento (ja incluida no
   repositorio):

   ```bash
   python -m app.build_index
   ```

4. Suba a API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Em outro terminal (com o passo 4 continuando a rodar), sirva o
   frontend (arquivos estaticos):

   ```bash
   cd frontend
   python3 -m http.server 5500
   ```

6. Acesse http://localhost:5500. O frontend/config.js já aponta para http://localhost:8000 por padrão, Ainda assim, é necessário garantir que a porta 8000 (backend) esteja configurada como pública e não como privada na aba PORTS do ambiente (clique com o botão direito na porta 8000 e selecione Port Visibility > Public), caso contrário o navegador não conseguirá se comunicar com a API mesmo com a URL correta configurada.

## Como executar no GitHub Codespaces

O projeto tambem pode ser executado dentro de um Codespace. Os passos
sao os mesmos das secoes anteriores (Docker ou execucao local), com
duas diferencas importantes causadas pela forma como o Codespaces expoe
portas para a internet:

1. Tornar a porta do backend publica. Na aba PORTS do Codespace,
   localize a porta 8000, clique com o botao direito nela e selecione
   Port Visibility > Public. Por padrao ela fica privada, o que impede
   o navegador de acessar a API.

2. Apontar o frontend para a URL publica do backend. Copie a URL
   publica gerada pelo Codespaces para a porta 8000 (formato
   `https://SEU-CODESPACE-8000.app.github.dev`, visivel na propria aba
   PORTS) e defina-a em `frontend/config.js`:

   ```js
   window.CHATBOT_API_BASE_URL = "https://SEU-CODESPACE-8000.app.github.dev";
   ```

   Isso e necessario porque o navegador acessa a aplicacao pela URL
   publica do Codespaces, nao pelo localhost da maquina que hospeda os
   containers ou processos.

   Se estiver usando Docker, reconstrua a imagem do frontend apos essa
   mudanca (ela e copiada para dentro da imagem no build):

   ```bash
   docker compose up --build frontend
   ```

   Se estiver rodando sem Docker (`python3 -m http.server`), basta
   salvar o arquivo e recarregar a pagina no navegador.

3. Atencao ao espaco em disco. Codespaces tem um limite de disco menor
   que uma maquina local. Se o build do Docker falhar por falta de
   espaco, limpe imagens e cache nao utilizados antes de tentar de
   novo:

   ```bash
   docker system prune -a -f
   ```

Fora essas duas diferencas, o restante do processo (configurar `.env`,
subir os containers ou processos, testar) e identico ao descrito nas
secoes anteriores.

## Regerando a base de conhecimento

O arquivo `filmes.json` ja vem pronto no repositorio, entao este passo
e opcional (util caso a equipe queira coletar uma base atualizada ou
com criterios diferentes).

1. Obtenha uma chave de API do TMDb (gratuita) em
   themoviedb.org/settings/api.

2. Instale a dependencia extra usada apenas por este script:

   ```bash
   cd backend
   source venv/bin/activate
   pip install -r data/requirements-dados.txt
   ```

3. Exporte sua chave do TMDb como variavel de ambiente (esta chave nao
   vai para o `.env` do projeto nem e usada pela aplicacao em runtime,
   apenas por este script pontual):

   ```bash
   export TMDB_API_KEY="sua_chave_do_tmdb_aqui"
   ```

4. Rode o script de coleta (leva cerca de 1 a 2 minutos, faz por volta
   de 200 requisicoes a API do TMDb):

   ```bash
   cd data
   python3 coletar_tmdb.py
   ```

   Isso sobrescreve `filmes.json` com uma nova coleta de 100 filmes.

5. Reindexe e, se estiver usando Docker, reconstrua a imagem do backend
   do zero para que o novo indice seja gerado (ver secao "Como executar
   com Docker", passo 6).

## Uso da interface

- Digite uma pergunta sobre algum dos 100 filmes da base: sinopse,
  diretor, elenco, genero, duracao, ano de lancamento, orcamento,
  bilheteria ou avaliacao.
- A conversa mantem memoria: perguntas de continuidade como "quem e o
  diretor do filme?" ou "qual a duracao?" (sem repetir o nome do filme)
  sao entendidas no contexto do filme mencionado anteriormente na mesma
  conversa.
- Cada resposta exibe a fonte (titulo do filme) que a LLM efetivamente
  citou para fundamentar a resposta. Quando o sistema nao encontra a
  informacao, nenhuma fonte e exibida.
- O botao "Nova conversa" limpa o historico da sessao atual (tanto no
  navegador quanto na memoria do backend) e inicia uma conversa nova.
- Perguntas fora do dominio da base (por exemplo, "qual a capital da
  Franca?") ou sem evidencia suficiente recebem uma resposta padrao
  informando que a informacao nao foi encontrada na base consultada,
  em vez de uma resposta inventada pela LLM.

## Decisoes e observacoes tecnicas

Esta secao documenta os principais problemas encontrados durante o
desenvolvimento e as solucoes adotadas, para referencia da equipe e de
quem for avaliar o projeto.


- Modelo da Groq: tentamos utilizar inicialmente o modelo
  `qwen/qwen3.6-27b`. Durante o desenvolvimento, esse modelo foi
  descontinuado pela Groq e passou a retornar erro 404 (model_not_found). O
  projeto foi atualizado para usar `qwen/qwen3.8-27b`.
- Chunking por fato, em linguagem natural (nao por contagem de
  palavras): a primeira versao do pipeline de indexacao cortava o
  texto de cada filme em blocos de tamanho fixo (~80 palavras), e cada
  fato era escrito como um rotulo tecnico ("Direcao: X. Duracao: Y.").
  Em testes com a base de 100 filmes (de estrutura textual repetitiva),
  esse formato causava recuperacoes incorretas: perguntas factuais
  como "quem dirigiu X?" muitas vezes recuperavam a sinopse de outro
  filme em vez da ficha tecnica correta, porque o modelo de embeddings
  (treinado sobre linguagem natural) associava melhor a pergunta a
  frases narrativas do que a blocos de rotulos concatenados. A solucao
  foi reescrever cada fato como uma frase natural e autocontida,
  repetindo o nome do filme dentro da propria frase (por exemplo, "O
  filme Cidade de Deus foi dirigido por Fernando Meirelles." em vez de
  "Direcao: Fernando Meirelles."), e gerar um chunk por fato em vez de
  cortar por contagem de palavras.
- Reescrita de pergunta antes da busca vetorial: perguntas de
  continuidade (por exemplo, "e quais premios ele ganhou?"), quando
  usadas diretamente na busca vetorial, geram embeddings pouco
  informativos e podem recuperar chunks do filme errado. Quando ha
  historico de conversa, a pergunta e reescrita por uma chamada a LLM
  antes da busca no FAISS, tornando-a autossuficiente.
- Filtragem de fontes citadas: a lista de fontes exibida na interface
  mostra apenas os chunks cuja numeracao [Fonte X] foi efetivamente
  citada pela LLM na resposta. Quando a LLM nao cita nenhuma fonte (por
  exemplo, na resposta de abstencao), a lista de fontes retornada e
  vazia, em vez de mostrar todos os chunks candidatos que foram
  recuperados mas descartados.
- PyTorch CPU-only: o Dockerfile do backend instala explicitamente a
  versao CPU-only do PyTorch (dependencia do sentence-transformers) a
  partir do indice oficial do PyTorch, evitando o download de
  dependencias de CUDA/GPU desnecessarias para este projeto, o que
  reduz significativamente o tamanho e o tempo de build da imagem.
- Modo offline do Hugging Face em runtime: o modelo de embeddings e
  baixado e cacheado durante o build da imagem Docker. Em tempo de
  execucao, o container roda com HF_HUB_OFFLINE=1 e
  TRANSFORMERS_OFFLINE=1, evitando que o container tente acessar a
  internet desnecessariamente ao iniciar.
- Logs estruturados do pipeline de RAG: cada etapa do grafo (pergunta
  original e reescrita, candidatos recuperados com seus scores, decisao
  do roteamento condicional, resposta final e fontes citadas) e
  registrada via logging, facilitando o diagnostico de problemas de
  recuperacao ou geracao diretamente pelos logs do backend (visiveis no
  terminal do uvicorn ou em docker compose up, sem -d).

## Limitacoes conhecidas

- A memoria de conversa e mantida em memoria de processo (nao em um
  banco de dados), associada a um session_id gerado pelo navegador.
  Isso e suficiente para o escopo desta atividade, mas significa que o
  historico de conversas e perdido caso o container do backend seja
  reiniciado.
- A conta gratuita da Groq tem limites de taxa (requisicoes e tokens
  por minuto). Em uso intenso e continuo, e possivel esbarrar
  ocasionalmente em erros 429, especialmente em perguntas de
  continuidade (que fazem duas chamadas a LLM: reescrita e geracao).
- Para alguns filmes, secoes especificas 
  podem, em casos pontuais, nao ser recuperadas entre os 8 candidatos
  considerados, caso a secao correta esteja competindo de perto com
  secoes de outros filmes na busca por similaridade. Isso e mitigado
  pelo TOP_K relativamente alto (8) e pela escrita das secoes em
  linguagem natural, mas nao eliminado por completo.
- A base de conhecimento cobre apenas os 100 filmes presentes em
  filmes.json (selecionados automaticamente por popularidade e nota no
  TMDb); perguntas sobre filmes fora dessa lista corretamente resultam
  na resposta de abstencao.
- Campos que a propria TMDb nao disponibiliza para um filme especifico
  (por exemplo, duracao ou orcamento de alguns titulos) sao registrados
  explicitamente como "nao informado" na base, e o assistente reporta
  essa ausencia em vez de inventar um valor.
