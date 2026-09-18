# Imagem base enxuta com Python.
FROM python:3.12-slim

# Evita geracao de arquivos .pyc e garante saida em tempo real dos logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala o PyTorch em versao CPU-only primeiro, usando o indice oficial
# do PyTorch para CPU. Isso evita que a instalacao padrao traga as
# dependencias de CUDA/GPU (pacotes nvidia-*), que sao desnecessarias
# neste projeto (rodamos apenas em CPU) e ocupam varios gigabytes,
# podendo esgotar o espaco em disco disponivel (por exemplo, em ambientes
# como o GitHub Codespaces).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Instala as demais dependencias, aproveitando o cache de camadas do
# Docker: elas so sao reinstaladas se requirements.txt mudar.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o codigo da aplicacao e a base de conhecimento.
COPY app ./app
COPY data ./data

# Gera o indice FAISS a partir da base de conhecimento durante o build da
# imagem. Neste passo o modelo de embeddings e baixado do Hugging Face
# Hub e fica cacheado dentro da imagem (em /root/.cache/huggingface).
RUN python -m app.build_index

# A partir daqui, forca o modo offline do Hugging Face Hub: o modelo de
# embeddings ja esta cacheado na imagem (passo acima), entao o container
# nao precisa nem deve tentar acessar a internet para checar
# atualizacoes do modelo ao iniciar. Isso evita falhas ou atrasos de
# inicializacao em ambientes onde o container tem rede restrita em
# tempo de execucao (mesmo que a rede tenha funcionado durante o build).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
