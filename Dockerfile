# Neuro-Sentinel Snake — Dockerfile
# Headless mode only (simulate + train + analyze + dashboard).
# Tkinter play mode requires a local Python install with display.

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data and model directories
RUN mkdir -p data models experiments

# Default: run the full pipeline then start dashboard
# Override with: docker run ... python main.py --mode simulate
EXPOSE 8501

CMD ["bash", "-c", \
     "python main.py --mode simulate --episodes 50 && \
      python main.py --mode train && \
      python main.py --mode analyze && \
      streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]