FROM python:3.10

WORKDIR /app

COPY auth-service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY auth-service/ .

COPY rl/ /app/rl/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]