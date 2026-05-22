FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


# Install liboqs
RUN git clone --depth=1 \
https://github.com/open-quantum-safe/liboqs.git

WORKDIR /app/liboqs

RUN mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    make install


ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /app


RUN pip uninstall -y oqs || true

RUN pip install --no-cache-dir \
git+https://github.com/open-quantum-safe/liboqs-python.git


# IMPORTANT FIX
COPY user-service/requirements.txt .

RUN pip install --no-cache-dir \
-r requirements.txt


# IMPORTANT FIX
COPY user-service/ .

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]