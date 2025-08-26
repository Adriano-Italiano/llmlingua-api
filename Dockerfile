FROM python:3.11-slim

WORKDIR /app

# Instalacja zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy pliki aplikacji
COPY . .

# Uruchamiamy serwer
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
