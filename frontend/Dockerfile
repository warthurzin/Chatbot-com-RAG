# Imagem enxuta do nginx para servir os arquivos estaticos do chat.
FROM nginx:1.27-alpine

# Remove a configuracao padrao do nginx para substituir pela nossa,
# que redireciona rotas desconhecidas para index.html (nao estritamente
# necessario para este projeto, mas e uma boa pratica para apps web).
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copia os arquivos estaticos do chat para o diretorio servido pelo nginx.
COPY index.html style.css script.js config.js /usr/share/nginx/html/

EXPOSE 80
