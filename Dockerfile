FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

#Data must be mounted at runtime via -v flag
#See README for data download instructions
VOLUME ["/app/data", "/app/outputs"]

CMD ["python", "main.py"]

