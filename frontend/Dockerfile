# Simple frontend Dockerfile (build + serve static files)
FROM node:20-alpine AS build
WORKDIR /app

# Build args for Vite env
ARG VITE_API_BASE
ENV VITE_API_BASE=${VITE_API_BASE:-/api}

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Basic nginx config for SPA
RUN printf 'server {\n  listen 8080;\n  server_name _;\n  root /usr/share/nginx/html;\n  index index.html;\n  location / { try_files $uri $uri/ /index.html; }\n}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
