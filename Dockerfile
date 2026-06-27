FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi==0.138.1 uvicorn==0.49.0 pydantic==2.13.4 \
    numpy==2.4.6 pandas==3.0.3 scikit-learn==1.9.0 joblib==1.5.3 \
    matplotlib==3.11.0 python-dateutil==2.9.0.post0 tzdata==2026.2

COPY notebooks/model.py notebooks/model.py
COPY src/ src/

EXPOSE 8080

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]