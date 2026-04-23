# 使用官方的 Python 3.11 轻量级基础镜像 (根据你之前的报错日志，你本地用的是 3.11)
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 Python 缓冲标准输出和生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统级依赖 (针对 PostgreSQL 连接库 psycopg2 及部分编译需求)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 把当前目录下的所有代码和配置复制到容器的 /app 目录中
COPY . /app/

# (注意：这里不需要写 CMD，因为我们在 docker-compose.yml 中已经分别为 api/worker/frontend 指定了不同的启动命令)