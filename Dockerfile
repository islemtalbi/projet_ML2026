FROM python:3.11-slim

# Métadonnées
LABEL maintainer="eco-smart-classifier"
LABEL description="Eco-Smart Classifier API — Classification de déchets"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    PORT=8000

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Téléchargement NLTK
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('punkt_tab', quiet=True)"

# Copie du code source
COPY src/ ./src/
COPY models/ ./models/

# Port exposé
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Lancement
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
