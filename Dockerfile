FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PORT=8000

WORKDIR $APP_HOME

# 系统依赖（CPU 版 Ultralytics 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY ultralytics ./ultralytics

# 先装较小的 CPU torch，再装其余依赖；pip 用清华源
RUN pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
      torch==2.1.0+cpu torchvision==0.16.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 运行时目录
RUN mkdir -p static/uploads static/predictions

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "app:app"]
